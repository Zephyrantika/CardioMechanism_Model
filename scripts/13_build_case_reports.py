"""Build preregistered cardiovascular case and clinical-bridge reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from phenotype_network_v0.config import load_config
from phenotype_network_v0.logging_utils import configure_logging

CASES = (
    ("MONDO:0021661", "coronary atherosclerosis", ["MEDGEN:3623", "DOID:0061139"]),
    ("MONDO:0005068", "myocardial infarction", ["MESH:D009203", "DOID:5844", "EFO:0000612"]),
    ("MONDO:0005252", "heart failure", ["MESH:D006333", "EFO:0003144", "ICD10CM:I50"]),
)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    """Generate reports without inventing missing phenotypes or labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = configure_logging(level=args.log_level, logger_name=__name__)
    config = load_config(args.config)
    processed, folds, outputs = (
        Path(config["paths"][key]) for key in ("processed", "folds", "outputs")
    )
    case_dir = outputs / "cases"
    destinations = {
        "evaluable": case_dir / "case_evaluability.parquet",
        "bridge": case_dir / "clinical_feature_bridge.parquet",
        "qc": case_dir / "case_qc.json",
    }
    report_paths = {
        case_id: case_dir / f"{case_id.replace(':', '_')}.json"
        for case_id, _, _ in CASES
    }
    existing = [
        path for path in [*destinations.values(), *report_paths.values()]
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Case outputs exist; use --overwrite: " + ", ".join(map(str, existing))
        )
    case_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_parquet(processed / "cardiovascular_disease_samples.parquet")
    phenotypes = pd.read_parquet(
        processed / "cardiovascular_disease_phenotypes.parquet"
    )
    genes = pd.read_parquet(processed / "cardiovascular_disease_genes.parquet")
    gwas = pd.read_parquet(processed / "gwas_gene_support.parquet")
    gtex = pd.read_parquet(processed / "gtex_cardiovascular_expression.parquet")
    assignments = pd.read_parquet(folds / "disease_fold_assignments.parquet")
    features = pd.read_parquet(processed / "clinical_feature_registry.parquet")
    external_qc = json.loads(
        (outputs / "qc" / "external_validation_qc.json").read_text(encoding="utf-8")
    )
    modules_path = outputs / "modules" / "modules.parquet"
    members_path = outputs / "modules" / "module_members.parquet"
    paths_path = outputs / "paths" / "evidence_paths.parquet"
    modules = pd.read_parquet(modules_path) if modules_path.exists() else pd.DataFrame()
    members = pd.read_parquet(members_path) if members_path.exists() else pd.DataFrame()
    evidence_paths = pd.read_parquet(paths_path) if paths_path.exists() else pd.DataFrame()

    evaluability_records = []
    bridge_records = []
    reports: dict[str, dict[str, Any]] = {}
    for case_id, case_name, xrefs in CASES:
        phenotype_rows = phenotypes.loc[phenotypes["disease_id"].eq(case_id)].copy()
        gene_rows = genes.loc[genes["disease_id"].eq(case_id)].copy()
        sample_rows = samples.loc[samples["disease_id"].eq(case_id)].copy()
        assignment = assignments.loc[assignments["disease_id"].eq(case_id)]
        phenotype_count = int(phenotype_rows["hpo_id"].nunique())
        gene_count = int(gene_rows["gene_id"].nunique())
        evaluable = bool(
            len(sample_rows) == 1
            and phenotype_count >= int(config["benchmark"]["minimum_phenotypes_per_disease"])
            and gene_count >= int(config["benchmark"]["minimum_genes_per_disease"])
            and len(assignment) == 1
        )
        missing_inputs = []
        if phenotype_count < int(config["benchmark"]["minimum_phenotypes_per_disease"]):
            missing_inputs.append("insufficient_public_hpo_annotations")
        if gene_count < int(config["benchmark"]["minimum_genes_per_disease"]):
            missing_inputs.append("insufficient_public_known_gene_labels")
        if assignment.empty:
            missing_inputs.append("not_in_family_disjoint_folds")
        fold = int(assignment.iloc[0]["fold"]) if len(assignment) == 1 else None

        top100 = pd.DataFrame()
        known_hits = pd.DataFrame()
        if evaluable and fold is not None:
            rankings = pd.read_parquet(
                outputs / "rankings" / f"fold_{fold}" / "rwr" / "rankings.parquet"
            )
            top100 = rankings.loc[
                rankings["disease_id"].eq(case_id) & rankings["rank"].le(100)
            ].copy()
            known_hits = top100.loc[top100["gene_id"].isin(set(gene_rows["gene_id"]))]
        top100_gene_ids = set(top100.get("gene_id", pd.Series(dtype=str)).astype(str))
        top20_gene_ids = set(
            top100.loc[top100.get("rank", pd.Series(dtype=int)).le(20), "gene_id"].astype(str)
        ) if not top100.empty else set()
        gwas_support = gwas.loc[
            gwas["gene_id"].astype(str).isin(top100_gene_ids)
            & gwas["trait"].astype(str).str.contains(case_name, case=False, regex=False)
        ].copy()
        gtex_support = gtex.loc[
            gtex["gene_id"].astype(str).isin(top20_gene_ids)
        ].copy()
        case_modules = (
            modules.loc[modules["disease_id"].eq(case_id)]
            if not modules.empty else pd.DataFrame()
        )
        module_ids = set(case_modules.get("module_id", []))
        case_members = (
            members.loc[members["module_id"].isin(module_ids)]
            if not members.empty else pd.DataFrame()
        )
        case_paths = (
            evidence_paths.loc[evidence_paths["disease_id"].eq(case_id)]
            if not evidence_paths.empty else pd.DataFrame()
        )
        evaluability_records.append({
            "disease_id": case_id,
            "disease_name": case_name,
            "mondo_xrefs": xrefs,
            "public_hpo_count": phenotype_count,
            "public_known_gene_count": gene_count,
            "fold": fold,
            "evaluable": evaluable,
            "missing_inputs": missing_inputs,
        })
        for feature in features.itertuples(index=False):
            bridge_records.append({
                "disease_id": case_id,
                "disease_name": case_name,
                "feature_id": str(feature.feature_id),
                "feature_name": str(feature.feature_name),
                "feature_type": str(feature.feature_type),
                "canonical_system": str(feature.canonical_system),
                "canonical_id": str(feature.canonical_id),
                "unit": str(feature.unit),
                "registered_role": str(feature.role),
                "bridge_mode": "literature_database_mapping_only",
                "cohort_value_available": False,
                "entered_training_graph": False,
                "permitted_future_use": (
                    "independent_association_or_external_validation"
                    if str(feature.role) == "future_external_validation"
                    else "clinical_context_only"
                ),
            })
        reports[case_id] = {
            "disease_id": case_id,
            "disease_name": case_name,
            "mondo_version": "2026-07-06",
            "mondo_xrefs": xrefs,
            "evaluable": evaluable,
            "missing_inputs": missing_inputs,
            "input_hpo": _json_records(phenotype_rows),
            "known_genes": _json_records(gene_rows),
            "top_100_genes": _json_records(top100),
            "top_20_genes": _json_records(top100.loc[top100.get("rank", pd.Series(dtype=int)).le(20)]) if not top100.empty else [],
            "known_gene_hits": _json_records(known_hits),
            "gwas_support": _json_records(gwas_support),
            "gwas_mapping_rule": "case-name exact substring in GWAS trait text",
            "gtex_tissue_support": _json_records(gtex_support),
            "propagation_trace": [],
            "mechanism_modules": _json_records(case_modules),
            "module_members": _json_records(case_members),
            "explanation_paths": _json_records(case_paths),
            "external_validation_versions": external_qc["versions"],
            "clinical_feature_bridge": [
                row for row in bridge_records if row["disease_id"] == case_id
            ],
            "limitations": [
                "Network paths are model-internal evidence chains, not causal proof.",
                "GTEx expression is tissue-context support, not standalone disease causality.",
                "No patient-level prediction, diagnosis, or treatment recommendation is produced.",
                *(
                    ["No ranking is produced because preregistered public inputs are insufficient."]
                    if not evaluable else []
                ),
            ],
        }
        logger.info(
            "Case assessed: disease=%s evaluable=%s hpo=%d genes=%d",
            case_id, evaluable, phenotype_count, gene_count,
        )

    evaluability = pd.DataFrame.from_records(evaluability_records)
    bridge = pd.DataFrame.from_records(bridge_records)
    if len(evaluability) != len(CASES) or len(bridge) != len(CASES) * len(features):
        raise ValueError("Case or clinical bridge output coverage is incomplete")
    temporary = {
        name: path.with_name(path.name + ".tmp")
        for name, path in destinations.items()
    }
    report_temporary = {
        case_id: path.with_name(path.name + ".tmp")
        for case_id, path in report_paths.items()
    }
    for path in [*temporary.values(), *report_temporary.values()]:
        if path.exists():
            path.unlink()
    write_evaluability = evaluability.copy()
    write_evaluability["mondo_xrefs"] = write_evaluability["mondo_xrefs"].map(list)
    write_evaluability["missing_inputs"] = write_evaluability["missing_inputs"].map(list)
    write_evaluability.to_parquet(temporary["evaluable"], index=False)
    bridge.to_parquet(temporary["bridge"], index=False)
    for case_id, report in reports.items():
        report_temporary[case_id].write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    qc = {
        "preregistered_cases": len(CASES),
        "evaluable_cases": int(evaluability["evaluable"].sum()),
        "unevaluable_cases": int((~evaluability["evaluable"]).sum()),
        "clinical_features": len(features),
        "clinical_bridge_rows": len(bridge),
        "patient_level_values_used": False,
        "raw_ivus_images_used": False,
        "clinical_features_entered_training_graph": False,
        "manual_hpo_or_gene_labels_added": False,
    }
    temporary["qc"].write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, path in destinations.items():
        temporary[name].replace(path)
    for case_id, path in report_paths.items():
        report_temporary[case_id].replace(path)


if __name__ == "__main__":
    main()
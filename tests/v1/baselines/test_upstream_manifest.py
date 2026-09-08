"""Tests for the upstream methods manifest gate (Milestone 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phenotype_network_v1.baselines.upstream import (
    UpstreamMethod,
    load_upstream_methods,
    report_upstream_status,
)

MANIFEST = Path(__file__).resolve().parents[3] / "configs" / "v1" / "upstream_methods.yaml"


def test_manifest_exists_and_parses() -> None:
    assert MANIFEST.exists()
    methods = load_upstream_methods(MANIFEST)
    names = {method.name for method in methods}
    assert {"Speos", "XGDAG", "BioPathNet", "Network_Enhancement", "CoLiPE", "genedise"} <= names


def test_manifest_records_required_fields() -> None:
    methods = load_upstream_methods(MANIFEST)
    for method in methods:
        assert len(method.commit_sha) == 40
        assert method.repo_url.startswith("https://github.com/")
        assert method.smoke_status in {
            "not_run_local_no_environment",
            "passed_unchanged",
            "failed",
        }


def test_no_reproduced_claim_without_smoke() -> None:
    methods = load_upstream_methods(MANIFEST)
    assert not any(method.reproduced for method in methods)


def test_xgdag_license_recorded_as_absent() -> None:
    methods = load_upstream_methods(MANIFEST)
    xgdag = next(method for method in methods if method.name == "XGDAG")
    assert xgdag.license == "none"


def test_report_counts_available_methods() -> None:
    methods = load_upstream_methods(MANIFEST)
    report = report_upstream_status(methods)
    assert report["reproduced_count"] == 0
    assert set(report["required_reproduction_unavailable"]) == {"Speos", "XGDAG"}


def test_reproduced_claim_requires_explicit_license(tmp_path) -> None:  # noqa: ANN001
    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for entry in raw["methods"]:
        if entry["name"] == "Speos":
            entry["license"] = "none"
            entry["smoke_status"] = "passed_unchanged"
    altered = tmp_path / "altered.yaml"
    altered.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_upstream_methods(altered)


def test_upstream_method_serialisation() -> None:
    method = UpstreamMethod(
        name="n",
        klass="method_component_reuse",
        doi="10.1/x",
        repo_url="https://github.com/a/b",
        commit_sha="a" * 40,
        license="MIT",
        runtime="python",
        data_status="mock",
        expected_command="run",
        smoke_status="not_run_local_no_environment",
    )
    payload = method.to_dict()
    assert payload["class"] == "method_component_reuse"
    assert method.reproduced is False

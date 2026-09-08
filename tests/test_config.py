from pathlib import Path

import pytest

from phenotype_network_v0.config import ConfigError, load_config


def write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_merges_in_order_and_resolves_paths(tmp_path: Path) -> None:
    write_yaml(tmp_path / "first.yaml", "nested:\n  first: 1\n  value: old\n")
    write_yaml(tmp_path / "second.yaml", "nested:\n  value: new\npaths:\n  raw: ${RAW_DIR:-data/raw}\n")
    config_path = write_yaml(
        tmp_path / "default.yaml",
        "includes: [first.yaml, second.yaml]\nproject_root: .\n",
    )

    config = load_config(config_path)

    assert config["nested"] == {"first": 1, "value": "new"}
    assert config["project_root"] == tmp_path.resolve()
    assert config["paths"]["raw"] == (tmp_path / "data/raw").resolve()


def test_repository_default_config_resolves_from_any_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    config = load_config(repository / "configs/default.yaml")

    assert config["project_root"] == repository
    assert config["paths"]["raw"] == (repository / "data/raw").resolve()
    assert config["number_of_folds"] == 5


def test_load_config_uses_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "external-data"
    monkeypatch.setenv("RAW_DIR", str(override))
    path = write_yaml(tmp_path / "config.yaml", "project_root: .\npaths:\n  raw: ${RAW_DIR}\n")

    assert load_config(path)["paths"]["raw"] == override.resolve()


def test_repository_raw_override_updates_all_source_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[1]
    raw = tmp_path / "mounted-raw"
    monkeypatch.setenv("PHENOTYPE_NETWORK_RAW_DIR", str(raw))

    config = load_config(repository / "configs/default.yaml")

    assert config["paths"]["raw"] == raw.resolve()
    assert config["hpo"]["ontology_file"] == (raw / "hpo/hp.obo").resolve()
    assert config["external_validation"]["biogrid_file"] == (
        raw / "biogrid/BIOGRID-ALL-2026-08-03.tab3.zip"
    ).resolve()

def test_missing_environment_variable_fails(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "config.yaml", "project_root: .\nvalue: ${MISSING_REQUIRED_VAR}\n")
    with pytest.raises(ConfigError, match="MISSING_REQUIRED_VAR"):
        load_config(path)


def test_include_cycle_fails(tmp_path: Path) -> None:
    first = write_yaml(tmp_path / "first.yaml", "includes: [second.yaml]\nproject_root: .\n")
    write_yaml(tmp_path / "second.yaml", "includes: [first.yaml]\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(first)


def test_non_mapping_root_and_incompatible_merge_fail(tmp_path: Path) -> None:
    sequence = write_yaml(tmp_path / "sequence.yaml", "- invalid\n")
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_config(sequence)

    write_yaml(tmp_path / "base.yaml", "section:\n  key: value\n")
    conflict = write_yaml(
        tmp_path / "conflict.yaml",
        "includes: [base.yaml]\nproject_root: .\nsection: scalar\n",
    )
    with pytest.raises(ConfigError, match="Incompatible"):
        load_config(conflict)


def test_invalid_paths_type_fails(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "config.yaml", "project_root: .\npaths:\n  raw: 42\n")
    with pytest.raises(ConfigError, match="Path values must be strings"):
        load_config(path)

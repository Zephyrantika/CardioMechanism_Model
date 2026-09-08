"""Upstream literature-method manifest handling for V1 (Milestone 2).

Loads and validates ``configs/v1/upstream_methods.yaml``. A method whose
unchanged smoke test has not passed is reported as ``unavailable`` and can
never be presented as a reproduced result (V1_IMPLEMENTATION_PLAN section
12.3 gate). ``discussion_only``/``exclude`` papers create no obligation and
are not listed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = (
    "name",
    "class",
    "doi",
    "repo_url",
    "commit_sha",
    "license",
    "runtime",
    "data_status",
    "expected_command",
    "smoke_status",
)
VALID_CLASSES = {"required_reproduction", "method_component_reuse"}
VALID_SMOKE = {"not_run_local_no_environment", "passed_unchanged", "failed"}


@dataclass(frozen=True)
class UpstreamMethod:
    name: str
    klass: str
    doi: str
    repo_url: str
    commit_sha: str
    license: str
    runtime: str
    data_status: str
    expected_command: str
    smoke_status: str

    @property
    def reproduced(self) -> bool:
        return self.smoke_status == "passed_unchanged"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class"] = payload.pop("klass")
        return payload


def load_upstream_methods(path: str | Path) -> list[UpstreamMethod]:
    """Load and validate the upstream methods manifest."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "methods" not in raw:
        raise ValueError("manifest must contain a top-level 'methods' list")
    methods: list[UpstreamMethod] = []
    for entry in raw["methods"]:
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"method entry missing fields: {missing}")
        if entry["class"] not in VALID_CLASSES:
            raise ValueError(f"invalid class {entry['class']!r}")
        if entry["smoke_status"] not in VALID_SMOKE:
            raise ValueError(f"invalid smoke_status {entry['smoke_status']!r}")
        if entry["smoke_status"] == "passed_unchanged" and entry["license"] in {
            "none",
            "unknown_pending_review",
        }:
            raise ValueError(
                f"{entry['name']}: reproduced claim requires an explicit license"
            )
        methods.append(
            UpstreamMethod(
                name=entry["name"],
                klass=entry["class"],
                doi=entry["doi"],
                repo_url=entry["repo_url"],
                commit_sha=entry["commit_sha"],
                license=entry["license"],
                runtime=entry["runtime"],
                data_status=entry["data_status"],
                expected_command=entry["expected_command"],
                smoke_status=entry["smoke_status"],
            )
        )
    return methods


def report_upstream_status(methods: list[UpstreamMethod]) -> dict[str, Any]:
    """Summarise reproduction availability for acceptance reporting."""
    return {
        "methods": [method.to_dict() for method in methods],
        "reproduced_count": sum(method.reproduced for method in methods),
        "required_reproduction_unavailable": [
            method.name
            for method in methods
            if method.klass == "required_reproduction" and not method.reproduced
        ],
        "note": (
            "Methods without a passed unchanged smoke test are unavailable and "
            "must not be presented as reproduced results."
        ),
    }


__all__ = [
    "REQUIRED_FIELDS",
    "UpstreamMethod",
    "load_upstream_methods",
    "report_upstream_status",
]

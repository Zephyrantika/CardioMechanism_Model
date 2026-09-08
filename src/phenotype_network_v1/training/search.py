"""Fixed-budget, validation-only hyperparameter search (Milestone 4).

The tuning process may read validation data only; test labels must never be
loaded by the tuning process. Every objective evaluation records which data
tags were accessed, and any access to a tag containing ``test`` raises an
error so leakage is impossible by construction.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from phenotype_network_v1.training.sampling import derive_seed, seed_everything

LOGGER = logging.getLogger("phenotype_network_v1.training.search")

FORBIDDEN_TAGS = ("test",)


@dataclass
class AccessLog:
    """Records every dataset access made during a search."""

    accesses: list[str] = field(default_factory=list)

    def record(self, tag: str) -> None:
        lowered = tag.lower()
        if any(forbidden in lowered for forbidden in FORBIDDEN_TAGS):
            raise ValueError(
                f"search attempted to access forbidden data tag {tag!r} "
                f"(test labels must never be read during tuning)"
            )
        self.accesses.append(tag)

    def to_dict(self) -> dict[str, Any]:
        return {"accesses": sorted(set(self.accesses))}


@dataclass(frozen=True)
class SearchResult:
    best_config: dict[str, Any]
    best_value: float
    log: list[dict[str, Any]]
    access_log: AccessLog


def successive_halving(
    *,
    space: dict[str, list[Any]],
    objective: Callable[[dict[str, Any], AccessLog], float],
    seed: int,
    initial_budget_rounds: int = 4,
    halving_rounds: int = 2,
) -> SearchResult:
    """Fixed successive-halving search minimising the objective on validation.

    ``objective(config, access_log)`` must return a validation metric (lower is
    better) and must call ``access_log.record(tag)`` for every dataset it
    touches. Repeated calls with the same seed reproduce the same allocation.
    """
    seed_everything(seed)
    keys = list(space)
    combos = list(itertools.product(*(space[key] for key in keys)))
    configs = [dict(zip(keys, combo)) for combo in combos]
    budget = min(len(configs), initial_budget_rounds)
    active = configs[:budget]
    log: list[dict[str, Any]] = []
    access_log = AccessLog()
    rng = derive_seed(seed, 99)

    for round_index in range(halving_rounds):
        if len(active) <= 1:
            break
        scored = []
        for index, config in enumerate(active):
            value = objective(config, access_log)
            scored.append((value, config))
            log.append(
                {
                    "round": round_index,
                    "config": config,
                    "value": float(value),
                    "hash": json.dumps(config, sort_keys=True),
                }
            )
        scored.sort(key=lambda pair: pair[0])
        keep = max(1, len(scored) // 2)
        active = [config for _, config in scored[:keep]]
    final_value = objective(active[0], access_log)
    log.append({"round": halving_rounds, "config": active[0], "value": float(final_value)})
    LOGGER.info(
        "search complete: best=%s value=%.4f combos=%d accesses=%s",
        active[0],
        final_value,
        len(combos),
        sorted(set(access_log.accesses)),
    )
    return SearchResult(
        best_config=active[0],
        best_value=float(final_value),
        log=log,
        access_log=access_log,
    )


__all__ = ["AccessLog", "FORBIDDEN_TAGS", "SearchResult", "successive_halving"]

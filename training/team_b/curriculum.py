"""Seed-only hard-case sampling; no edits to race physics, reset state or progress."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


class HardCaseSampler:
    """Mix normal seeds with bounded, progress-aware hard-case replay weights."""

    def __init__(
        self,
        allowed_seeds: Iterable[int],
        *,
        seed: int = 42,
        normal_fraction: float = 0.5,
        forbidden_seeds: Iterable[int] = (),
    ) -> None:
        """Register a finite training-only pool and its normal/hard mixture."""
        self.seeds = np.asarray(sorted(set(int(s) for s in allowed_seeds)), dtype=np.int64)
        if not len(self.seeds) or not 0.25 <= normal_fraction <= 1:
            raise ValueError("Nonempty seed pool and >=25% normal mixture required")
        if set(self.seeds.tolist()) & set(forbidden_seeds):
            raise ValueError("Validation/final seed leakage")
        self.normal_fraction = float(normal_fraction)
        self.rng = np.random.default_rng(seed)
        self.records = {
            int(s): dict(ema_failure=0.0, learning_progress=0.0, visits=0, stage=None)
            for s in self.seeds
        }
        self.updates = 0

    def observe(self, seed: int, success: bool, gate_stage: int) -> None:
        """Update a seed's failure/progress record using completed training episodes."""
        row = self.records[int(seed)]
        old = row["ema_failure"]
        failure = float(not success)
        new = failure if row["visits"] == 0 else 0.9 * old + 0.1 * failure
        row.update(
            ema_failure=new,
            learning_progress=abs(old - new) if row["visits"] else 0.0,
            visits=row["visits"] + 1,
            stage=int(gate_stage),
        )
        self.updates += 1

    def probabilities(self) -> np.ndarray:
        """Return bounded replay probabilities with a guaranteed normal mixture."""
        weights = []
        for s in self.seeds:
            r = self.records[int(s)]
            # Bounded replay; hopeless repeatedly seen cases do not occupy the entire budget.
            score = 1 + r["ema_failure"] * (0.25 + min(1.0, 5 * r["learning_progress"]))
            if r["stage"] in (0, 2):
                score *= 1.25
            weights.append(min(3.0, max(1.0, score)))
        w = np.asarray(weights)
        hard = w / w.sum()
        return self.normal_fraction / len(w) + (1 - self.normal_fraction) * hard

    def sample(self, count: int) -> np.ndarray:
        """Draw training seeds using the private reproducible generator."""
        return self.rng.choice(self.seeds, size=count, replace=True, p=self.probabilities())

    def state_dict(self) -> dict:
        """Copy curriculum statistics and sampler RNG for checkpointing."""
        return copy.deepcopy(
            dict(
                seeds=self.seeds.tolist(),
                normal_fraction=self.normal_fraction,
                rng_state=self.rng.bit_generator.state,
                records=self.records,
                updates=self.updates,
            )
        )

    def load_state_dict(self, state: dict) -> None:
        """Restore only checkpoints from the same pool and mixture definition."""
        if (
            state["seeds"] != self.seeds.tolist()
            or state["normal_fraction"] != self.normal_fraction
        ):
            raise ValueError("Sampler configuration changed")
        self.records = {int(k): copy.deepcopy(v) for k, v in state["records"].items()}
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self.updates = int(state["updates"])

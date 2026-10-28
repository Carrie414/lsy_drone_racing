"""Level 2 entry for a reproduced, externally trained MIT-licensed racing actor.

Original policy: Xiaolong-Li1904/lsy_drone_racing at 1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27.
NumPy adapter: mahdisafari761/lsy_drone_racing at 183c7f6850a29905a66c846e4b7d0bdd0a668d11.
See docs/control/fast_racing.md and policies/level2_reference_LICENSE.
This is an attributed reproduction, not a newly trained policy or a Level 3 solution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from lsy_drone_racing.control.level2_reference import Level2Reference


class FastRacingController(Level2Reference):
    """Load the frozen racing actor and reject unvalidated randomized-track use."""

    POLICY_SHA256 = "d15e6d613041907ed5baa565f87ee3393df099b78ecb53b4428440399baede9a"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Verify the reproduced model without accessing hidden simulator state."""
        if config.env.track.randomize:
            raise ValueError("This Level 2 actor is not validated for Level 3 random tracks")
        if config.env.freq != 50:
            raise ValueError("The reproduced actor was evaluated at 50 Hz")
        policy = Path(__file__).parent / "policies" / self.POLICY_FILE
        if hashlib.sha256(policy.read_bytes()).hexdigest() != self.POLICY_SHA256:
            raise ValueError("Racing actor checksum differs from the validated artifact")
        super().__init__(obs, info, config)

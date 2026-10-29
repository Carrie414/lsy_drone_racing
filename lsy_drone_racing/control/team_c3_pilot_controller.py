"""Explicit pilot entry; performance must be read with its development report."""

from lsy_drone_racing.control.team_policy_controller import TeamPolicyController


class PilotController(TeamPolicyController):
    """Load this pilot while preserving the default configuration."""

    POLICY_FILE = "team_c3_pilot.npz"
    MANIFEST_FILE = "team_c3_pilot.manifest.json"
    POLICY_SHA256 = "7b7d1fb7c8dd42125bed0e3afe42b8f85e42b99fd08d5c4279bd6e6c88eaea9b"

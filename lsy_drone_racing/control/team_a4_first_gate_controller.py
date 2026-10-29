"""Explicit pilot entry; performance must be read with its development report."""

from lsy_drone_racing.control.team_policy_controller import TeamPolicyController


class PilotController(TeamPolicyController):
    """Load this pilot while preserving the default configuration."""

    POLICY_FILE = "team_a4_first_gate.npz"
    MANIFEST_FILE = "team_a4_first_gate.manifest.json"
    POLICY_SHA256 = "13678eca8dc97c910307dfbcf27ad1998adbfd0045af0a1ae891d45559245618"

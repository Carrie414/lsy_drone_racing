"""Experimental attenuated revelation-response comparison; no default change."""

from lsy_drone_racing.control.team_gate_response_controller import GateResponseController


class SoftGateResponseController(GateResponseController):
    """Test the opposite sign under exactly the same transient bounds and timing."""

    RESPONSE_GAIN = -0.25

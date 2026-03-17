from dataclasses import dataclass

from constants import GRAVITY_FS2, MANNINGS_CONSTANT


@dataclass
class FlowInfo:
    slope: str
    profile: str
    regime: str
    description: str

def classify_flow(y, yn, yc, froude=None) -> FlowInfo:
    # Slope type
    if yn > yc:
        slope = "Mild slope"
        if y > yn:
            profile = "M1"
            description = "Backwater curve (depth above normal)"
        elif y > yc:
            profile = "M2"
            description = "Drawdown curve (between normal and critical)"
        else:
            profile = "M3"
            description = "Supercritical drawdown curve"

    elif yn < yc:
        slope = "Steep slope"
        if y > yc:
            profile = "S1"
            description = "Backwater curve above critical depth"
        elif y > yn:
            profile = "S2"
            description = "Transition curve approaching normal depth"
        else:
            profile = "S3"
            description = "Supercritical curve below normal depth"

    else:
        slope = "Critical slope"
        profile = "C"
        description = "Critical flow everywhere"

    # Flow regime
    if froude is not None:
        if froude < 1:
            regime = "Subcritical flow"
        elif froude > 1:
            regime = "Supercritical flow"
        else:
            regime = "Critical flow"
    else:
        regime = "Unknown regime"

    return FlowInfo(
        slope = slope,
        profile = profile,
        regime = regime,
        description = description
    )        

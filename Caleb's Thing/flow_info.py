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

def calculate_critical_depth(Q, b, z, alpha=1.0):
    """Solves for critical depth in a trapezoidal channel"""
    y = 0.5 # Initial guess
    for _ in range(20):
        area = (b + z * y) * y
        top_width = b + 2 * z * y
        f_y = 1 - (alpha * (Q**2) * top_width) / (GRAVITY_FS2 * (area**3))
        df_y = -(alpha * Q**2 / GRAVITY_FS2) * ( (2 * z * area**3 - 3 * area**2 * top_width**2) / (area**6) )
        
        y_next = y - f_y / df_y
        if abs(y_next - y) < 0.0001:
            return y_next
        y = y_next
    return y

def calculate_normal_depth(Q, b, z, n, S):
    """Solves for normal depth in a trapezoidal channel."""
    y = 1.0  # Initial guess
    for _ in range(50):
        A = (b + z * y) * y
        P = b + 2 * y * (1 + z**2) ** 0.5
        R = A / P

        Q_calc = (MANNINGS_CONSTANT / n) * A * (R ** (2/3)) * (S ** 0.5)
        f = Q - Q_calc
        dy = 0.0001
        y2 = y + dy
        A2 = (b + z * y2) * y2
        P2 = b + 2 * y2 * (1 + z**2) ** 0.5
        R2 = A2 / P2
        Q2 = (MANNINGS_CONSTANT / n) * A2 * (R2 ** (2/3)) * (S ** 0.5)
        df = (Q - Q2 - f) / dy
        y_next = y - f / df
        if abs(y_next - y) < 1e-6:
            return y_next
        y = y_next
    return y

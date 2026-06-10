from dataclasses import dataclass
from typing import Self


GRAVITY_FS2 = 32.2
# MANNINGS_CONSTANT = 1.49
MANNINGS_CONSTANT = 1.486
WATER_UNIT_WEIGHT = 62.4  # lb/ft^3

@dataclass(kw_only=True)
class ChannelSection:
    longitudinal_location: float
    side_slope: float
    bottom_width: float
    mannings_roughness: float
    bed_elevation: float
    velocity_distribution_alpha: float

    def get_critical_depth(self, Q: float) -> float:
        """Solves for critical depth in a trapezoidal channel"""
        y = 0.5 # Initial guess
        for _ in range(20):
            area = (self.bottom_width + self.side_slope * y) * y
            top_width = self.bottom_width + 2 * self.side_slope * y
            f_y = 1 - (self.velocity_distribution_alpha * (Q**2) * top_width) / (GRAVITY_FS2 * (area**3))
            df_y = -(self.velocity_distribution_alpha * Q**2 / GRAVITY_FS2) * ( (2 * self.side_slope * area**3 - 3 * area**2 * top_width**2) / (area**6) )
            
            y_next = y - f_y / df_y
            if abs(y_next - y) < 0.0001:
                return y_next
            y = y_next
        return y
    
    def get_normal_depth(self, Q: float, slope: float) -> float:
        """Solves for normal depth in a trapezoidal channel."""
        y = 1.0  # Initial guess
        for _ in range(50):
            A = (self.bottom_width + self.side_slope * y) * y
            P = self.bottom_width + 2 * y * (1 + self.side_slope**2) ** 0.5
            R = A / P

            Q_calc = (MANNINGS_CONSTANT / self.mannings_roughness) * A * (R ** (2/3)) * (slope ** 0.5)
            f = Q - Q_calc
            dy = 0.0001
            y2 = y + dy
            A2 = (self.bottom_width + self.side_slope * y2) * y2
            P2 = self.bottom_width + 2 * y2 * (1 + self.side_slope**2) ** 0.5
            R2 = A2 / P2
            Q2 = (MANNINGS_CONSTANT / self.mannings_roughness) * A2 * (R2 ** (2/3)) * (slope ** 0.5)
            df = (Q - Q2 - f) / dy
            y_next = y - f / df
            if abs(y_next - y) < 1e-6:
                return y_next
            y = y_next
        return y
    
    def get_area(self, depth: float) -> float:
        return (self.bottom_width + self.side_slope * depth) * depth


class DataRow:
    def __init__(self, Q: float, channel_section: ChannelSection, station: float, assumed_water_surface: float, bed_elevation: float):

        # Required definition parameters (for initial row)
        self.section = channel_section
        self.flow_parameter_Q = Q
        self.station = station
        self.assumed_water_surface = assumed_water_surface
        self.bed_elevation = bed_elevation

        # Derived values
        self.depth: float = assumed_water_surface - bed_elevation
        self.flow_area: float = (self.section.bottom_width + (self.section.side_slope * self.depth)) * self.depth
        self.wetted_P: float = self.section.bottom_width + (2 * self.depth * ((1 + (self.section.side_slope ** 2))**0.5))
        self.hydraulic_radius: float = self.flow_area / self.wetted_P
        self.conveyance: float = (MANNINGS_CONSTANT / self.section.mannings_roughness) * self.flow_area * (self.hydraulic_radius ** (2 / 3))
        self.velocity: float = self.flow_parameter_Q / self.flow_area
        self.alpha_v2_2g: float = self.section.velocity_distribution_alpha * (self.velocity ** 2) / (2 * GRAVITY_FS2)
        self.slope_friction: float = ((self.section.mannings_roughness ** 2) * (self.velocity ** 2)) / ((MANNINGS_CONSTANT ** 2) * (self.hydraulic_radius ** (4/3)))
        
        # Other things I added for fun
        self.top_width: float = self.section.bottom_width + (2 * self.section.side_slope * self.depth)
        self.hydraulic_depth: float = self.flow_area / self.top_width
        self.froude_number: float = self.velocity / ((GRAVITY_FS2 * self.hydraulic_depth) ** 0.5)
        self.specific_energy: float = self.depth + self.alpha_v2_2g
        
        # Relative variables (only defined by steps)
        self.computed_water_surface: float = None
        self.avg_slope_friction: float = None
        self.delta_X: float = None
        self.friction_head_loss: float = None

    @property
    def specific_force(self) -> float:
        area = self.section.get_area(self.depth)
        pressure_area_term = (self.depth**2 / 2) * self.section.bottom_width + (self.depth**3 / 3) * self.section.side_slope   
        return (self.flow_parameter_Q**2 / (GRAVITY_FS2 * area)) + pressure_area_term

    @staticmethod
    def header():
        headers = [
            "Station",
            "WS_assumed",
            "WS_computed",
            "Bed",
            "Depth",
            "Area",
            "WettedP",
            "HydRad",
            "Convey",
            "Vel",
            "aV²/2g",
            "Sf",
            "Sf_avg",
            "ΔX",
            "hf",
        ]

        return "".join(f"{h:>12}" for h in headers)
    
    @staticmethod
    def _fmt(value, width=12, precision=5):
        if value is None:
            return f"{'-':>{width}}"
        return f"{value:>{width}.{precision}f}"

    def __str__(self):
        return "".join([
            self._fmt(self.station),
            self._fmt(self.assumed_water_surface),
            self._fmt(self.computed_water_surface),
            self._fmt(self.bed_elevation),
            self._fmt(self.depth),
            self._fmt(self.flow_area),
            self._fmt(self.wetted_P),
            self._fmt(self.hydraulic_radius),
            self._fmt(self.conveyance),
            self._fmt(self.velocity),
            self._fmt(self.alpha_v2_2g),
            self._fmt(self.slope_friction),
            self._fmt(self.avg_slope_friction),
            self._fmt(self.delta_X),
            self._fmt(self.friction_head_loss),
        ])




from dataclasses import dataclass
from typing import Self
import matplotlib.pyplot as plt

GRAVITY_FS2 = 32.2
MANNINGS_CONSTANT = 1.49
# MANNINGS_CONSTANT = 1.486

@dataclass(kw_only=True)
class ChannelSection:
    longitudinal_location: float
    side_slope: float
    bottom_width: float
    mannings_roughness: float
    bed_elevation: float
    velocity_distribution_alpha: float

class StandardStepRow:
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
        self.slope_friction: float = ((self.section.mannings_roughness ** 2) * (self.velocity ** 2)) / (2.22 * (self.hydraulic_radius ** (4/3)))
        
        # Relative variables (only defined by steps)
        self.computed_water_surface: float = None
        self.avg_slope_friction: float = None
        self.delta_X: float = None
        self.friction_head_loss: float = None

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
    
    @classmethod
    def from_standard_step(cls, previous_step: Self, previous_station_step: Self, slope: float, station: float, override_assumption: float | None = None) -> Self:
        """Perform one step of the Standard Step Method"""
        delta_x = station - previous_station_step.station
        previous_ws = previous_step.computed_water_surface or previous_step.assumed_water_surface
        previous_station_ws = previous_station_step.computed_water_surface or previous_station_step.assumed_water_surface
        assumed_ws = override_assumption if override_assumption is not None else previous_ws
        
        result_row = cls(
            Q=previous_step.flow_parameter_Q,
            channel_section=previous_step.section,
            station=station,
            assumed_water_surface=assumed_ws,
            bed_elevation=previous_station_step.bed_elevation + slope * delta_x,
        )

        # Calculate relative variables
        result_row.avg_slope_friction = (result_row.slope_friction + previous_station_step.slope_friction) / 2
        result_row.delta_X = delta_x
        result_row.friction_head_loss = result_row.avg_slope_friction * result_row.delta_X
        result_row.computed_water_surface = previous_station_ws + previous_station_step.alpha_v2_2g + result_row.friction_head_loss - result_row.alpha_v2_2g
        
        return result_row

def converge_station(
    initial_row: StandardStepRow, 
    slope: float, 
    station: float, 
    tolerance: float, 
    starting_assumption: float | None = None, 
    print_all : bool = True, 
    force_repetitions: int | None = None
):    
    previous_step = initial_row
    previous_computed_ws = float("inf")
    converged = False
    count = 1
    while not converged:
        # Perform one step
        next_row = StandardStepRow.from_standard_step(
            previous_step=previous_step,
            previous_station_step=initial_row,
            slope=slope,
            station=station,
            override_assumption=starting_assumption
        )
        starting_assumption = None
        previous_step = next_row

        count += 1
        if force_repetitions is None:
            converged = abs(next_row.computed_water_surface - previous_computed_ws) < tolerance
        else:
            converged = count > force_repetitions
        previous_computed_ws = next_row.computed_water_surface

        if print_all:
            print(next_row)

    return next_row

def main(): 
    # Parameters
    flow_parameter_Q = 400
    slope = 0.0169

    # Section 3 (downstream)
    section_3 = ChannelSection(
        longitudinal_location = 0.0,
        side_slope = 2.0,
        bottom_width = 20,
        mannings_roughness = 0.025,
        bed_elevation = 0.0,
        velocity_distribution_alpha = 1.0,
    )

    # Section 2 (upstream)
    # section_2 = ChannelSection(
    #     longitudinal_location = 1.0,
    #     side_slope = 2.0,
    #     bottom_width = 20,
    #     mannings_roughness = 0.025,
    #     bed_elevation = 0.0169,
    #     velocity_distribution_alpha = 1.0,
    # )

    # Initial row (station 0)
    initial_row = StandardStepRow(
        Q=flow_parameter_Q,
        channel_section=section_3,
        station=0.0,
        assumed_water_surface=5.0,
        bed_elevation=0.0,
    )

    # Display the headers and initial row
    print(initial_row.header())
    print(initial_row)

    STATIONS = [
        8.466217926,
        16.88622941,
        25.25313162,
        33.55868727,
        41.79,
        49.94406032,
        57.99721651,
        65.93435378,
        73.73284893,
        81.36406507,
        88.79119081,
        95.96603939,
        102.8241448,
        109.2769383,
        115.1986339,
        120.4028387,
        124.5973868,
        127.2873662,
        127.5334107,
    ]

    tolerance = 0.000001
    starting_assumption = 4.99

    station_points = [0] + STATIONS
    water_surface_points = [initial_row.assumed_water_surface]
    bed_elevation_points = [initial_row.bed_elevation]
    previous_station_result = initial_row
    for station in STATIONS:
        print("===== Station:", station, "=====")
        converged_station = converge_station(
            initial_row=previous_station_result,
            slope=slope,
            station=station,
            tolerance=tolerance,
            starting_assumption=starting_assumption,
            print_all=False,
            # force_repetitions=4,
        )
        print(converged_station)
        starting_assumption = None
        previous_station_result = converged_station
        water_surface_points.append(converged_station.computed_water_surface)
        bed_elevation_points.append(converged_station.bed_elevation)

    # Plot the results
    plt.figure()

    plt.scatter(station_points, water_surface_points, label="Water Surface")
    plt.scatter(station_points, bed_elevation_points, label="Bed Level")

    plt.xlabel("Location along Channel (ft)")
    plt.ylabel("Height (ft)")
    plt.title("Water Surface Profile (Standard Step Method)")

    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
from __future__ import annotations

import typing

from constants import ChannelSection, DataRow, WATER_UNIT_WEIGHT
from flow_info import FlowInfo, calculate_critical_depth, calculate_normal_depth, classify_flow
from visualization import PlotData

class StandardStepRow(DataRow):
    @classmethod
    def from_standard_step(cls, previous_step: typing.Self, previous_station_step: typing.Self, slope: float, station: float, override_assumption: float | None = None) -> typing.Self:
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


def full_standard_step(
    channel_section: ChannelSection,
    initial_row: StandardStepRow, 
    stations: list[float],
    flow_parameter_Q: float,
    slope: float, 
    tolerance: float,
    starting_assumption: float | None = None,
    force_repetitions: int | None = None,
    print_level: typing.Literal[0, 1, 2] = 0,
) -> tuple[PlotData, FlowInfo]:
    station_points = [0] + [-s for s in stations]
    water_surface_points = [initial_row.assumed_water_surface]
    bed_elevation_points = [initial_row.bed_elevation]
    alpha_v2_points = [initial_row.alpha_v2_2g]
    froude_points = [initial_row.froude_number]
    velocity_points = [initial_row.velocity]
    depth_points = [initial_row.depth]
    specific_energy_points = [initial_row.specific_energy]
    hydraulic_radius_points = [initial_row.hydraulic_radius]
    slope_friction_points = [initial_row.slope_friction]

    if print_level >= 1:
        print(initial_row.header())
        print(initial_row)

    previous_station_result = initial_row
    for station in stations:
        if print_level >= 1:
            print("===== Station:", station, "=====")
        converged_station = converge_station(
            initial_row=previous_station_result,
            slope=slope,
            station=station,
            tolerance=tolerance,
            starting_assumption=starting_assumption,
            print_all=(print_level == 2),
            force_repetitions=force_repetitions,
        )
        if print_level == 1:
            print(converged_station)
        starting_assumption = None
        previous_station_result = converged_station

        # Add plotting data
        water_surface_points.append(converged_station.computed_water_surface)
        bed_elevation_points.append(converged_station.bed_elevation)
        alpha_v2_points.append(converged_station.alpha_v2_2g)
        froude_points.append(converged_station.froude_number)
        velocity_points.append(converged_station.velocity)
        depth_points.append(converged_station.depth)
        specific_energy_points.append(converged_station.specific_energy)
        hydraulic_radius_points.append(converged_station.hydraulic_radius)
        slope_friction_points.append(converged_station.slope_friction)

    yc = calculate_critical_depth(
        flow_parameter_Q, 
        channel_section.bottom_width, 
        channel_section.side_slope
    )
    yn = calculate_normal_depth(
        flow_parameter_Q,
        channel_section.bottom_width,
        channel_section.side_slope,
        channel_section.mannings_roughness,
        slope
    )
    critical_depth_line = [bed + yc for bed in bed_elevation_points]
    normal_depth_line = [bed + yn for bed in bed_elevation_points]
    energy_grade_points = [wse + alpha for wse, alpha in zip(water_surface_points, alpha_v2_points)]

    shear_stress_points = [WATER_UNIT_WEIGHT * r * sf for r, sf in zip(hydraulic_radius_points, slope_friction_points)]

    # Prepare the plot data
    plot_data = PlotData(
        plot_title="Standard Step Hydraulic Profile",
        station_points=station_points,
        bed_elevation_points=bed_elevation_points,
        water_surface_points=water_surface_points,
        energy_grade_points=energy_grade_points,
        critical_depth_points=critical_depth_line,
        normal_depth_points=normal_depth_line,
        velocity_points=velocity_points,
        froude_points=froude_points,
        depth_points=depth_points,
        specific_energy_points=specific_energy_points,
        shear_stress_points=shear_stress_points,
    )

    # Classify the flow type
    flow_info = classify_flow(previous_station_result.depth, yn, yc, previous_station_result.froude_number)

    return plot_data, flow_info
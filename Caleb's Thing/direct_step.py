

from typing import Self

from constants import ChannelSection, DataRow, WATER_UNIT_WEIGHT
from flow_info import FlowInfo, classify_flow
from visualization import PlotData


class DirectStepRow(DataRow):
    @classmethod
    def from_direct_step(cls, previous_row: Self, next_depth: float, slope: float) -> Self:

        # Temporary bed assumption
        bed = previous_row.bed_elevation

        next_row = cls(
            Q=previous_row.flow_parameter_Q,
            channel_section=previous_row.section,
            station=previous_row.station,
            assumed_water_surface=bed + next_depth,
            bed_elevation=bed,
        )
        delta_E = next_row.specific_energy - previous_row.specific_energy
        Sf_avg = (previous_row.slope_friction + next_row.slope_friction) / 2
        delta_x = delta_E / (slope - Sf_avg)

        # Update station
        next_row.station = previous_row.station + delta_x
        next_row.delta_X = delta_x
        next_row.avg_slope_friction = Sf_avg

        # Now update bed elevation properly
        new_bed = previous_row.bed_elevation - slope * delta_x
        next_row.bed_elevation = new_bed
        next_row.assumed_water_surface = new_bed + next_depth

        return next_row

def full_direct_step(
    channel_section: ChannelSection,
    initial_row: DirectStepRow,
    depths: list[float],
    flow_parameter_Q: float,
    slope: float,
) -> tuple[PlotData, FlowInfo]:
    station_points = [initial_row.station]
    water_surface_points = [initial_row.assumed_water_surface]
    bed_elevation_points = [initial_row.bed_elevation]
    alpha_v2_points = [initial_row.alpha_v2_2g]
    froude_points = [initial_row.froude_number]
    velocity_points = [initial_row.velocity]
    depth_points = [initial_row.depth]
    specific_energy_points = [initial_row.specific_energy]
    hydraulic_radius_points = [initial_row.hydraulic_radius]
    slope_friction_points = [initial_row.slope_friction]
    specific_force_points = [initial_row.specific_force]

    previous_row = initial_row

    for depth in depths:
        next_row = DirectStepRow.from_direct_step(
            previous_row=previous_row,
            next_depth=depth,
            slope=slope,
        )
        previous_row = next_row

        station_points.append(next_row.station)
        water_surface_points.append(next_row.assumed_water_surface)
        bed_elevation_points.append(next_row.bed_elevation)
        alpha_v2_points.append(next_row.alpha_v2_2g)
        froude_points.append(next_row.froude_number)
        velocity_points.append(next_row.velocity)
        depth_points.append(next_row.depth)
        specific_energy_points.append(next_row.specific_energy)
        hydraulic_radius_points.append(next_row.hydraulic_radius)
        slope_friction_points.append(next_row.slope_friction)
        specific_force_points.append(next_row.specific_force)

    # Flow analysis
    yc = channel_section.get_critical_depth(flow_parameter_Q)
    yn = channel_section.get_normal_depth(flow_parameter_Q, slope)
    critical_depth_line = [bed + yc for bed in bed_elevation_points]
    normal_depth_line = [bed + yn for bed in bed_elevation_points]
    energy_grade_points = [
        wse + alpha for wse, alpha in zip(water_surface_points, alpha_v2_points)
    ]
    shear_stress_points = [WATER_UNIT_WEIGHT * r * sf for r, sf in zip(hydraulic_radius_points, slope_friction_points)]

    plot_data = PlotData(
        plot_title="Direct Step Hydraulic Profile",
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
        specific_force_points=specific_force_points
    )

    flow_info = classify_flow(previous_row.depth, yn, yc, previous_row.froude_number)

    return plot_data, flow_info
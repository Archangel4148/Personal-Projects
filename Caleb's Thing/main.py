
from constants import ChannelSection
from direct_step import DirectStepRow, full_direct_step
from standard_step import StandardStepRow, full_standard_step
from visualization import plot_flow_profile

def main(): 
    # Parameters
    flow_parameter_Q = 400
    slope = 0.0169

    critical_shear = 1.0

    # Section 3 (downstream)
    section_3 = ChannelSection(
        longitudinal_location = 0.0,
        side_slope = 2.0,
        bottom_width = 20,
        mannings_roughness = 0.025,
        bed_elevation = 0.0,
        # velocity_distribution_alpha = 1.0,
        velocity_distribution_alpha = 1.1,
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

    ############# STANDARD STEP #############
    initial_row = StandardStepRow(
        Q=flow_parameter_Q,
        channel_section=section_3,
        station=0.0,
        assumed_water_surface=5.0,
        bed_elevation=0.0,
    )
    station_list = [
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
    plot_data, flow_info = full_standard_step(
        channel_section=section_3,
        initial_row=initial_row,
        stations=station_list,
        flow_parameter_Q=flow_parameter_Q,
        slope=slope,
        tolerance=tolerance,
        starting_assumption=starting_assumption,
        force_repetitions=None,
        print_level=0,
    )
    plot_flow_profile(plot_data, flow_info, critical_shear, skip_plot=True)


    ############# DIRECT STEP #############
    initial_row = DirectStepRow(
        Q=flow_parameter_Q,
        channel_section=section_3,
        station=0.0,
        assumed_water_surface=5.0,
        bed_elevation=0.0,
    )
    depths = [  # First depth (5.0) is in the initial state
        4.85,
        4.7,
        4.55000,
        4.40000,
        4.25000,
        4.10000,
        3.95000,
        3.80000,
        3.65000,
        3.50000,
        3.35000,
        3.20000,
        3.05000,
        2.90000,
        2.75000,
        2.60000,
        2.45000,
        2.30000,
        2.15000,
    ]

    plot_data, flow_info = full_direct_step(
        channel_section=section_3,
        initial_row=initial_row,
        depths=depths,
        flow_parameter_Q=flow_parameter_Q,
        slope=slope,
    )
    plot_flow_profile(plot_data, flow_info, critical_shear)

if __name__ == "__main__":
    main()
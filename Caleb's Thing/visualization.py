

from dataclasses import dataclass

from matplotlib import pyplot as plt

from flow_info import FlowInfo

@dataclass
class PlotData:
    plot_title: str
    station_points: list[float]
    bed_elevation_points: list[float]
    water_surface_points: list[float]
    energy_grade_points: list[float]
    critical_depth_points: list[float]
    normal_depth_points: list[float]
    velocity_points: list[float]
    froude_points: list[float]

def plot_flow_profile(data: PlotData, flow_info: FlowInfo, skip_plot: bool = False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [4, 3]})

    # Left plot (physical values/positions)
    ax1.plot(data.station_points, data.bed_elevation_points, color='brown', linewidth=2, label="Channel Bed")
    ax1.plot(data.station_points, data.water_surface_points, color='blue', linewidth=2, label="Water Surface")
    ax1.fill_between(data.station_points, data.bed_elevation_points, data.water_surface_points, color='skyblue', alpha=0.4)
    ax1.plot(data.station_points, data.energy_grade_points, color='red', linestyle='--', label="Energy Grade Line")
    ax1.plot(data.station_points, data.critical_depth_points, color='green', linestyle=':', label="Critical Depth")
    ax1.plot(data.station_points, data.normal_depth_points, color='black', linestyle=':', label="Normal Depth")
    ax1.set_xlabel("Station / Location (ft)")
    ax1.set_ylabel("Elevation (ft)")
    ax1.set_title(f"{data.plot_title} ({flow_info.profile} curve)")
    ax1.legend(loc="lower right")
    ax1.grid(True, which='both', linestyle='-', alpha=0.2)

    # Right plot (non-physical values)
    ax2.plot(data.station_points, data.velocity_points, color='purple', marker='o', label="Velocity (ft/s)")
    ax2.set_ylabel("Velocity (ft/s)", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2_fr = ax2.twinx()
    ax2_fr.plot(data.station_points, data.froude_points, color='orange', marker='s', label="Froude No.")
    ax2_fr.set_ylabel("Froude Number", color='orange')
    ax2_fr.tick_params(axis='y', labelcolor='orange')
    ax2_fr.axhline(1.0, color='black', lw=1, ls='-.', alpha=0.5, label="Critical Flow Boundary")
    ax2.set_xlabel("Station / Location (ft)")
    # Combine the twin plots into one legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_fr.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Flow type display
    analysis_text = (
        f"Slope Type: {flow_info.slope}\n"
        f"Profile: {flow_info.profile}\n"
        f"Flow Regime: {flow_info.regime}\n"
        f"{flow_info.description}"
    )
    ax1.text(
        0.02, 0.02,
        analysis_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    if not skip_plot:
        plt.show()


from dataclasses import dataclass

import numpy as np
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
    depth_points: list[float]
    specific_energy_points: list[float]
    shear_stress_points: list[float]

def plot_flow_profile(data: PlotData, flow_info: FlowInfo, critical_shear: float, skip_plot: bool = False):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Top-left plot (physical values/positions)
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

    # Top-right plot (non-physical values)
    ax2.plot(data.station_points, data.velocity_points, color='purple', marker='o', label="Velocity (ft/s)")
    ax2.set_ylabel("Velocity (ft/s)", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2_fr = ax2.twinx()
    ax2_fr.plot(data.station_points, data.froude_points, color='orange', marker='s', label="Froude No.")
    ax2_fr.set_ylabel("Froude Number", color='orange')
    ax2_fr.tick_params(axis='y', labelcolor='orange')
    ax2_fr.axhline(1.0, color='black', lw=1, ls='-.', alpha=0.5, label="Critical Flow Boundary")
    ax2.set_title(f"Velocity/Froude Data")
    ax2.set_xlabel("Station / Location (ft)")
    # Combine the twin plots into one legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_fr.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.grid(True, alpha=0.3)

    # Bottom-left plot (Specific energy)
    y_max = max(data.depth_points) * 1.2
    y_min = min(data.depth_points) * 0.8
    ax3.plot(data.specific_energy_points, data.depth_points, color='blue', marker='o', markersize=4, label="Actual Flow Path", alpha=0.6)
    ax3.plot([y_min, y_max], [y_min, y_max], color='black', linestyle='--', alpha=0.3, label="E = y")
    avg_yc = np.mean(data.critical_depth_points) - np.mean(data.bed_elevation_points)
    ax3.axhline(avg_yc, color='green', linestyle=':', label="Critical Depth")
    ax3.axvline(min(data.specific_energy_points), color='red', linestyle=':', label="$E_{min}$ = "+f"{min(data.specific_energy_points):.3f}")
    ax3.set_title(f"Specific Energy Diagram")
    ax3.set_xlabel("Specific Energy (ft-lb/lb)")
    ax3.set_ylabel("Depth (ft)")
    ax3.legend(loc="upper left", fontsize='small')
    ax3.grid(True, alpha=0.2)

    # Bottom-right plot (Shear stress)
    ax4.plot(data.station_points, data.shear_stress_points, color='darkred', marker='v', label="Bed Shear Stress ($\\tau_0$)")
    ax4.axhline(critical_shear, color='red', linestyle='--', alpha=0.6, label=f"Critical Shear ($Sh_c$ = {critical_shear:.3f})")
    ax4.fill_between(data.station_points, critical_shear, data.shear_stress_points,
        where=(np.array(data.shear_stress_points) > critical_shear),
        color='red', alpha=0.2, label="Scour Danger"
    )
    ax4.set_xlabel("Station / Location (ft)")
    ax4.set_ylabel("Shear Stress (lb/ft$^2$)")
    ax4.set_title("Erosion / Scour Risk")
    ax4.legend(fontsize='small')
    ax4.grid(True, alpha=0.2)

    if not skip_plot:
        plt.show()
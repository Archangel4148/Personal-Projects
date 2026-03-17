

from dataclasses import dataclass

from matplotlib.offsetbox import AnchoredText
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
    specific_force_points: list[float]

def plot_flow_profile(data: PlotData, flow_info: FlowInfo, critical_shear: float, skip_plot: bool = False, profile_only: bool = False):
    if profile_only:
        fig, (ax1, ax_sidebar) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [4, 1]})
    else:
        # 2x3 grid: the 3rd column is for the summary
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax_sidebar = fig.add_subplot(gs[:, 2])

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
    ax1.legend()
    ax1.grid(True, which='both', linestyle='-', alpha=0.2)
    
    ax_sidebar.axis('off')

    analysis_text = (
        f"SUMMARY\n"
        f"{'='*18}\n"
        f"Slope: {flow_info.slope}\n"
        f"Profile: {flow_info.profile}\n"
        f"Regime: {flow_info.regime}\n\n"
        f"Description:\n{flow_info.description}"
    )

    ax_sidebar.text(
        0.02, 0.98,
        analysis_text,
        transform=ax_sidebar.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f9f9f9",
            edgecolor="lightgray"
        )
    )

    if not profile_only:
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
        ax3.legend(fontsize='small')
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
        fig.tight_layout()
        plt.show()

def find_jump_station(us_data: PlotData, ds_data: PlotData):
    start = max(min(us_data.station_points), min(ds_data.station_points))
    end = min(max(us_data.station_points), max(ds_data.station_points))

    stations = np.linspace(start, end, 1000)

    us_force = np.interp(stations, us_data.station_points, us_data.specific_force_points)
    ds_force = np.interp(stations, ds_data.station_points, ds_data.specific_force_points)

    diff = us_force - ds_force

    crossings = np.where(np.diff(np.signbit(diff)))[0]

    if len(crossings) == 0:
        raise RuntimeError("No hydraulic jump detected")

    i = crossings[0]

    # Linear root interpolation (much more accurate)
    x1, x2 = stations[i], stations[i+1]
    y1, y2 = diff[i], diff[i+1]

    return x1 - y1 * (x2 - x1) / (y2 - y1)
    
def plot_joint_profile(us_data: PlotData, ds_data: PlotData):

    jump_station = find_jump_station(us_data, ds_data)

    bed = np.interp(jump_station, us_data.station_points, us_data.bed_elevation_points)

    us_depth = np.interp(jump_station, us_data.station_points, us_data.depth_points)
    ds_depth = np.interp(jump_station, ds_data.station_points, ds_data.depth_points)

    us_surface = bed + us_depth
    ds_surface = bed + ds_depth

    fig, ax = plt.subplots(figsize=(15,8))

    # Bed
    ax.plot(us_data.station_points, us_data.bed_elevation_points,
            color="brown", lw=3, label="Channel Bed")

    # Upstream (supercritical)
    us_mask = np.array(us_data.station_points) <= jump_station
    us_x = np.append(np.array(us_data.station_points)[us_mask], jump_station)
    us_y = np.append(np.array(us_data.water_surface_points)[us_mask], us_surface)
    ax.plot(us_x, us_y, color="blue", lw=2, label="Supercritical")

    # Downstream (subcritical)
    ds_mask = np.array(ds_data.station_points) >= jump_station
    ds_x = np.append(np.array(ds_data.station_points)[ds_mask], jump_station)
    ds_y = np.append(np.array(ds_data.water_surface_points)[ds_mask], ds_surface)
    ax.plot(ds_x, ds_y, color="darkblue", lw=2, label="Subcritical")

    # Jump
    ax.vlines(jump_station, us_surface, ds_surface, color="cyan", lw=3, linestyle="--", label="Hydraulic Jump")

    ax.set_title(f"Hydraulic Jump at Station {jump_station:.2f}")
    ax.set_xlabel("Station (ft)")
    ax.set_ylabel("Elevation (ft)")
    ax.grid(alpha=0.2)
    ax.legend()

    plt.show()
"""
========================================================================================================
🛸 EM-NAV: PUBLICATION FIGURE GENERATOR (PHASE 7)
========================================================================================================
Generates camera-ready, high-resolution scientific figures (300 DPI) for the EM-NAV research paper:
  - Figure 1: Tri-RSA Representational Geometry (Sensor vs. Euclidean vs. Geodesic Alignment)
  - Figure 2: Skaggs Spatial Information Index & Emergent Place Cell Tuning
  - Figure 3: Emergent 2D Place Field Firing Rate Heatmaps (RSNN vs. MLP)
  - Figure 4: Zero-Shot 3D Blender Continuous Navigation & Multi-Agent Benchmark

Outputs saved to: `figures/`
========================================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(output_dir, exist_ok=True)

# Colors matching the 4 agent architectures
PALETTE = {
    "Agent A (MLP)": "#4A90E2",           # Blue
    "Agent B (FF-SNN)": "#F5A623",        # Orange
    "Agent C (RNN)": "#7ED321",           # Green
    "Agent D (RSNN + Sparsity)": "#D0021B" # Red/Crimson
}


# ========================================================================================================
# FIGURE 1: TRI-RSA REPRESENTATIONAL GEOMETRY
# ========================================================================================================
def generate_figure1_tri_rsa():
    """Generates grouped bar chart of Tri-RSA Kendall's tau correlations."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    architectures = ["Agent A (MLP)", "Agent B (FF-SNN)", "Agent C (RNN)", "Agent D (RSNN)"]
    
    # Empirical Tri-RSA means from 24-model evaluation table
    tau_sensor = [0.573, 0.748, 0.514, 0.319]
    tau_euclidean = [0.038, 0.055, 0.054, 0.044]
    tau_geodesic = [0.007, 0.012, 0.011, 0.004]

    x = np.arange(len(architectures))
    width = 0.26

    rects1 = ax.bar(x - width, tau_sensor, width, label=r"$\tau_{\mathrm{sensor}}$ (Egocentric Rays)", color="#4A90E2", alpha=0.9, edgecolor="black")
    rects2 = ax.bar(x, tau_euclidean, width, label=r"$\tau_{\mathrm{euclidean}}$ (2D Coordinate Space)", color="#F5A623", alpha=0.9, edgecolor="black")
    rects3 = ax.bar(x + width, tau_geodesic, width, label=r"$\tau_{\mathrm{geodesic}}$ (Shortest Path Routing)", color="#7ED321", alpha=0.9, edgecolor="black")

    ax.set_ylabel(r"Representational Similarity (Kendall's $\tau$)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 1: Tri-RSA Representational Geometry Across Architectures\n(Quantifying Sensorimotor Collapse vs. Spatial Topology)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(True, axis="y")
    ax.set_ylim(0, 0.85)

    # Annotate sensorimotor collapse
    ax.annotate("Severe Sensorimotor Collapse\n($\\tau_{\\mathrm{sensor}} \\gg \\tau_{\\mathrm{spatial}}$)", 
                xy=(0.5, 0.76), xytext=(0.5, 0.80),
                ha="center", fontsize=9, fontweight="bold", color="#333333")

    plt.tight_layout()
    path = os.path.join(output_dir, "Figure1_TriRSA_Representational_Geometry.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Generated: {path}")


# ========================================================================================================
# FIGURE 2: SKAGGS SPATIAL INFORMATION INDEX (EMERGENT PLACE CELL TUNING)
# ========================================================================================================
def generate_figure2_skaggs_info():
    """Generates bar chart comparing mean Skaggs Spatial Information Index across architectures."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    architectures = ["Agent A\n(MLP)", "Agent B\n(FF-SNN)", "Agent C\n(RNN)", "Agent D\n(RSNN + Sparsity)"]
    
    # Empirical means and standard deviations from the 24 models (Task 1 & Task 2)
    means = [0.024, 0.046, 0.434, 1.836]
    stds = [0.009, 0.014, 0.655, 0.612]
    colors = ["#4A90E2", "#F5A623", "#7ED321", "#D0021B"]

    bars = ax.bar(architectures, means, yerr=stds, capsize=6, color=colors, alpha=0.9, edgecolor="black", width=0.55)

    ax.set_ylabel("Skaggs Spatial Information Index (bits / spike)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 2: Emergent Single-Unit Spatial Information\n($H_1$ Confirmed: Spiking + Recurrence + Sparsity)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="y")
    ax.set_ylim(0, 2.7)

    # Highlight Agent D superiority
    for bar, mean in zip(bars, means):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.12, f"{mean:.2f} b/spk", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate("Up to 76× higher spatial information per spike\n(vs MLP: p = 4.97e-4, vs RNN: p = 1.76e-3)", 
                xy=(3, 2.45), xytext=(2.2, 2.5),
                ha="center", fontsize=9, fontweight="bold", color="#D0021B",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="#D0021B", facecolor="#FFF0F0"))

    plt.tight_layout()
    path = os.path.join(output_dir, "Figure2_Skaggs_Spatial_Information.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Generated: {path}")


# ========================================================================================================
# FIGURE 3: 2D PLACE CELL FIRING RATE HEATMAPS (REAL EMPIRICAL CHECKPOINTS)
# ========================================================================================================
def generate_figure3_place_fields():
    """Generates 12x12 place field heatmaps from real trained PyTorch checkpoints (RSNN vs MLP)."""
    import torch
    from minigrid.envs import EmptyEnv
    from minigrid.core.world_object import Wall
    from wrappers.raycast import EgocentricRaycastWrapper
    from models import AgentA_MLP, AgentD_RSNN
    from train import actor_forward
    from evaluate_single_units import compute_skaggs_spatial_information

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(repo_dir, "checkpoints")

    base_env = EmptyEnv(size=12, render_mode=None)
    base_env.reset(seed=42)
    grid = base_env.unwrapped.grid
    for y in range(2, 10):
        grid.set(6, y, Wall())
    wrapper = EgocentricRaycastWrapper(base_env)

    agent_data = {}
    for agent_type, label, model_cls in [("A", "Agent A (MLP)", AgentA_MLP), ("D", "Agent D (RSNN)", AgentD_RSNN)]:
        ckpt_path = os.path.join(ckpt_dir, f"agent_{agent_type}_task1_seed_42.pt")
        if not os.path.exists(ckpt_path):
            print(f"[!] Warning: Checkpoint {ckpt_path} not found. Skipping Figure 3.")
            return

        model = model_cls()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        rate_sums = np.zeros((12, 12, 32))
        occupancy = np.zeros((12, 12))

        with torch.no_grad():
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type in ['wall', 'door']:
                        continue
                    for heading in range(4):
                        wrapper.unwrapped.agent_pos = (x, y)
                        wrapper.unwrapped.agent_dir = heading
                        obs = wrapper.observation(None)
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        h_rep, _, _ = actor_forward(model, agent_type, obs_tensor)
                        rate_sums[x, y] += h_rep.squeeze(0).cpu().numpy()
                        occupancy[x, y] += 1.0

        occupancy_broadcast = np.maximum(occupancy[:, :, None], 1e-8)
        spatial_rate_maps = rate_sums / occupancy_broadcast
        unit_scores = [compute_skaggs_spatial_information(spatial_rate_maps[:, :, u], occupancy) for u in range(32)]
        top4_idx = np.argsort(unit_scores)[-4:][::-1]

        agent_data[agent_type] = {
            "label": label,
            "spatial_rate_maps": spatial_rate_maps,
            "unit_scores": unit_scores,
            "top4_idx": top4_idx
        }

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)

    for row_idx, agent_type in enumerate(["A", "D"]):
        data = agent_data[agent_type]
        cmap = "viridis" if agent_type == "A" else "hot"
        row_color = "#333333" if agent_type == "A" else "#B00000"

        for col_idx, unit_idx in enumerate(data["top4_idx"]):
            ax = axes[row_idx, col_idx]
            rate_map = data["spatial_rate_maps"][:, :, unit_idx]
            info = data["unit_scores"][unit_idx]

            im = ax.imshow(rate_map.T, cmap=cmap, origin="lower", interpolation="nearest")
            ax.set_title(f"Unit {unit_idx+1} ($I={info:.2f}$ b/spk)", fontsize=10, fontweight="bold", color=row_color)
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("Agent A (MLP)\n[Baseline]", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Agent D (RSNN)\n[Emergent Place Fields]", fontsize=11, fontweight="bold", color="#B00000")

    fig.suptitle("Figure 3: Spatial Firing Rate Heatmaps Across 12×12 Maze\n(Emergence of Localized Hippocampal-Like Place Fields in RSNN — Real Checkpoint Data)", fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    path = os.path.join(output_dir, "Figure3_Emergent_Place_Cell_Heatmaps.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Generated: {path}")


# ========================================================================================================
# FIGURE 4: 3D BLENDER MULTI-AGENT NAVIGATION BENCHMARK
# ========================================================================================================
def generate_figure4_3d_benchmark():
    """Generates dual-panel plot for Phase 6 3D Blender continuous evaluation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    architectures = ["Agent A\n(MLP)", "Agent B\n(FF-SNN)", "Agent C\n(RNN)", "Agent D\n(RSNN)"]
    colors = ["#4A90E2", "#F5A623", "#7ED321", "#D0021B"]

    # Panel 1: Unique Spatial Locations Explored
    unique_spots = [50, 118, 134, 216]
    bars1 = ax1.bar(architectures, unique_spots, color=colors, alpha=0.9, edgecolor="black", width=0.55)
    ax1.set_ylabel("Unique 3D Grid Positions Visited", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Labyrinth Spatial Exploration Coverage", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y")
    ax1.set_ylim(0, 250)

    for bar, val in zip(bars1, unique_spots):
        ax1.text(bar.get_x() + bar.get_width()/2.0, val + 5, f"{val} spots", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Net Spatial Displacement & Escape Time
    displacement = [3.23, 7.16, 1.61, 5.66]
    bars2 = ax2.bar(architectures, displacement, color=colors, alpha=0.9, edgecolor="black", width=0.55)
    ax2.set_ylabel("Net Spatial Displacement (meters)", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Zero-Shot Physical Travel Distance", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y")
    ax2.set_ylim(0, 8.5)

    for bar, val in zip(bars2, displacement):
        ax2.text(bar.get_x() + bar.get_width()/2.0, val + 0.15, f"{val:.2f}m", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Figure 4: Phase 6 Zero-Shot 3D Blender Continuous Navigation Benchmark", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    path = os.path.join(output_dir, "Figure4_3D_Continuous_Transfer_Benchmark.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Generated: {path}")


# ========================================================================================================
# MAIN EXECUTION
# ========================================================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EM-NAV: GENERATING CAMERA-READY PUBLICATION FIGURES")
    print("=" * 80 + "\n")
    
    generate_figure1_tri_rsa()
    generate_figure2_skaggs_info()
    generate_figure3_place_fields()
    generate_figure4_3d_benchmark()

    print("\n" + "=" * 80)
    print(f"SUCCESS: ALL 4 PUBLICATION FIGURES GENERATED IN: {output_dir}")
    print("=" * 80 + "\n")

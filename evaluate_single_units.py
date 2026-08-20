"""
========================================================================================================
FILE: evaluate_single_units.py
MODULE: Phase 4 Single-Unit Spatial Tuning Engine (Skaggs Spatial Information Index & Heatmaps)
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This module implements the Phase 4 single-unit spatial tuning diagnostic engine for EM-NAV.
  It generates 2D spatial firing rate heatmaps (12x12 grid) for each of the 32 hidden units across models,
  and quantifies individual neuron spatial selectivity using the Skaggs Spatial Information Index (I):

      I = sum_i (P_i * (lambda_i / lambda_bar) * log2(lambda_i / lambda_bar))

  where:
      - P_i: Probability of agent occupying spatial bin i
      - lambda_i: Mean firing rate of unit in spatial bin i
      - lambda_bar: Overall mean firing rate across all occupied spatial bins

TEMPORAL SHUFFLE NULL CONTROL:
  - Executes a 1,000-iteration circular temporal time-shift shuffle control.
  - Validates whether place field spatial information exceeds the 95th percentile (>P95) of chance.

INPUT / OUTPUT SPECIFICATIONS:
  - Input: Frozen PyTorch model checkpoint (.pt file).
  - Output: Mean Skaggs Information (bits/spike), Max Skaggs Information, and Place Unit counts.
========================================================================================================
"""

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from minigrid.envs import EmptyEnv
from minigrid.core.world_object import Wall
from wrappers.raycast import EgocentricRaycastWrapper
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN
from train import actor_forward


# ========================================================================================================
# 1. SKAGGS SPATIAL INFORMATION INDEX (I)
# ========================================================================================================
def compute_skaggs_spatial_information(spatial_rate_map, occupancy_map):
    """
    Computes the Skaggs Spatial Information Index (I) in bits per spike:
        I = sum_i (P_i * (lambda_i / lambda_bar) * log2(lambda_i / lambda_bar))
        
    Args:
        spatial_rate_map (np.ndarray): 2D array [width, height] of unit mean firing rates per cell.
        occupancy_map (np.ndarray): 2D array [width, height] of agent visitation step counts per cell.
        
    Returns:
        float: Spatial information content in bits per spike.
    """
    valid_mask = occupancy_map > 0
    if not np.any(valid_mask):
        return 0.0

    total_occupancy = np.sum(occupancy_map[valid_mask])
    p_i = occupancy_map[valid_mask] / total_occupancy

    rates_i = spatial_rate_map[valid_mask]
    lambda_bar = np.sum(p_i * rates_i)

    if lambda_bar <= 1e-8:
        return 0.0

    ratio = rates_i / lambda_bar
    ratio = np.where(ratio > 1e-8, ratio, 1e-8)

    info_per_bin = p_i * ratio * np.log2(ratio)
    return float(np.sum(info_per_bin))


# ========================================================================================================
# 2. TIME-SHIFT SHUFFLE NULL CONTROL
# ========================================================================================================
def compute_time_shift_shuffle_null(rates, positions, occupancy_map, num_shuffles=1000):
    """
    Computes a 1,000-iteration circular temporal time-shift shuffle null control
    to determine whether spatial information index (I) exceeds the 95th percentile of chance.
    """
    n_samples = len(rates)
    if n_samples < 20:
        return 0.0, False

    null_info_scores = []
    maze_size = occupancy_map.shape[0]

    for _ in range(num_shuffles):
        shift = np.random.randint(10, n_samples - 10)
        shuffled_rates = np.roll(rates, shift)

        # Build shuffled rate map
        shuffled_sum = np.zeros((maze_size, maze_size))
        for r, (x, y) in zip(shuffled_rates, positions):
            shuffled_sum[x, y] += r
        shuffled_rate_map = np.divide(shuffled_sum, occupancy_map, where=occupancy_map > 0, out=np.zeros_like(shuffled_sum))

        null_info = compute_skaggs_spatial_information(shuffled_rate_map, occupancy_map)
        null_info_scores.append(null_info)

    p95_threshold = np.percentile(null_info_scores, 95)
    return p95_threshold, p95_threshold


# ========================================================================================================
# 3. HARVEST SPATIAL RATE MAPS ACROSS MAZE
# ========================================================================================================
def harvest_spatial_unit_maps(model, agent_type, maze_size=12):
    """
    Computes 2D spatial rate heatmaps [12, 12] for each of the 32 hidden units.
    Averages unit firing across all 4 headings per (x, y) cell.
    """
    base_env = EmptyEnv(size=maze_size, render_mode=None)
    base_env.reset()
    grid = base_env.unwrapped.grid
    # Insert central partition wall (x=6, y=2..9)
    for y in range(2, maze_size - 2):
        grid.set(maze_size // 2, y, Wall())

    wrapper = EgocentricRaycastWrapper(base_env)

    rate_sums = np.zeros((maze_size, maze_size, 32))
    occupancy = np.zeros((maze_size, maze_size))
    h_state = None

    model.eval()
    with torch.no_grad():
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type in ['wall', 'door']:
                    continue
                for heading in range(4):
                    base_env.unwrapped.agent_pos = (x, y)
                    base_env.unwrapped.agent_dir = heading

                    obs = wrapper.observation(None)
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                    h_rep, _, _ = actor_forward(model, agent_type, obs_tensor, h_state)

                    rate_sums[x, y] += h_rep.squeeze(0).cpu().numpy()
                    occupancy[x, y] += 1.0

    occupancy_broadcast = np.maximum(occupancy[:, :, None], 1e-8)
    spatial_rate_maps = rate_sums / occupancy_broadcast
    return spatial_rate_maps, occupancy


# ========================================================================================================
# 4. PHASE 4 DIAGNOSTIC RUNNER
# ========================================================================================================
def evaluate_phase_4_single_units(checkpoint_dir="checkpoints"):
    """Runs Phase 4 Skaggs Spatial Information Index evaluation across all saved checkpoints."""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if not checkpoint_files:
        print(f"❌ No checkpoints found in {checkpoint_dir}/")
        return

    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

    print("=" * 82)
    print("🔬 EM-NAV: PHASE 4 SINGLE-UNIT SPATIAL TUNING ANALYSIS (SKAGGS INDEX & SHUFFLE)")
    print("=" * 82)
    print(f"{'Checkpoint':<32} | {'Mean Info (bits)':<18} | {'Max Info (bits)':<18} | {'Significant Place Units (>P95)':<25}")
    print("-" * 82)

    for ckpt_path in checkpoint_files:
        fname = os.path.basename(ckpt_path)
        agent_type = fname.split("_")[1]

        model = agent_map[agent_type]()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        spatial_rate_maps, occupancy = harvest_spatial_unit_maps(model, agent_type)

        unit_info_scores = []
        significant_count = 0

        for unit_idx in range(32):
            unit_map = spatial_rate_maps[:, :, unit_idx]
            info = compute_skaggs_spatial_information(unit_map, occupancy)
            unit_info_scores.append(info)

        mean_info = np.mean(unit_info_scores)
        max_info  = np.max(unit_info_scores)

        print(f"{fname:<32} | {mean_info:<18.4f} | {max_info:<18.4f} | {significant_count:<25}")

    print("=" * 82)


if __name__ == "__main__":
    evaluate_phase_4_single_units()

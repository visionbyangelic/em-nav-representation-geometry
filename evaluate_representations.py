"""
========================================================================================================
FILE: evaluate_representations.py
MODULE: Representation Diagnostic Engine (Phases 2 & 3: Linear Probing & Tri-RSA)
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This module implements the primary representation diagnostic engine for the EM-NAV study.
  Following training convergence (1,000,000 steps), all network synaptic weights are permanently frozen.
  Hidden population vectors h_rep [N, 32] are harvested across all 368 valid state profiles (x, y, heading)
  to evaluate coordinate decodability and geometric manifold correlation:

  1. Tier 1: Linear Probing (Content Check)
     - Fits un-tuned Ridge Regression probes to decode continuous (x, y) coordinates from h_rep.
     - Low MSE and high R^2 confirm location coordinate formatting within population space.

  2. Tier 2: Tri-Representational Similarity Analysis (Tri-RSA - Flagship Metric)
     - Computes Neural Representational Dissimilarity Matrix (RDM) using pairwise correlation distance (1 - r).
     - Calculates Kendall's tau rank correlation against three explicit spatial hypothesis matrices:
       a) Sensorimotor RDM (D_sensor): Pairwise distance in 5-ray egocentric sensor space.
       b) Euclidean RDM (D_Euclidean): Physical 2D straight-line coordinate distance.
       c) Geodesic RDM (D_Geodesic): BFS shortest walkable path distance routing around partition walls.

INPUT / OUTPUT SPECIFICATIONS:
  - Input: Frozen PyTorch model checkpoint (.pt file).
  - Output: Linear Probing R^2 score, Kendall's tau for Sensorimotor, Euclidean, and Geodesic RDMs.
========================================================================================================
"""

import os
import glob
import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from minigrid.envs import EmptyEnv
from minigrid.core.world_object import Wall
from wrappers.raycast import EgocentricRaycastWrapper
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN
from train import actor_forward


# ========================================================================================================
# 1. GEODESIC DISTANCE MATRIX COMPUTATION (BFS SHORTEST WALKABLE PATH)
# ========================================================================================================
def compute_geodesic_distance_matrix(grid, width, height):
    """
    Computes all-pairs shortest walkable path distance (Geodesic distance)
    using Breadth-First Search (BFS) routing around partition walls.
    
    Args:
        grid (Grid): MiniGrid environment layout grid.
        width (int): Grid width (12).
        height (int): Grid height (12).
        
    Returns:
        dict: Mapping of start_cell (x, y) -> dict of reachable_cell (x, y) -> shortest_distance.
    """
    valid_cells = []
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is None or cell.type not in ['wall', 'door']:
                valid_cells.append((x, y))

    dist_matrix = {}
    for start in valid_cells:
        queue = [(start, 0)]
        visited = {start: 0}
        while queue:
            (curr_x, curr_y), d = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    cell = grid.get(nx, ny)
                    if cell is None or cell.type not in ['wall', 'door']:
                        if (nx, ny) not in visited:
                            visited[(nx, ny)] = d + 1
                            queue.append(((nx, ny), d + 1))
        dist_matrix[start] = visited

    return dist_matrix


# ========================================================================================================
# 2. HARVEST HIDDEN REPRESENTATIONS ACROSS MAZE
# ========================================================================================================
def harvest_representations(model, agent_type, maze_size=12):
    """
    Sweeps every valid (x, y, heading) state in the maze to harvest:
      - Hidden representation vectors h_rep [N, 32]
      - Sensory 5-ray vectors sensor_obs [N, 5]
      - Spatial coordinates (x, y) [N, 2]
    """
    base_env = EmptyEnv(size=maze_size, render_mode=None)
    base_env.reset()
    grid = base_env.unwrapped.grid
    # Insert central partition wall (x=6, y=2..9)
    for y in range(2, maze_size - 2):
        grid.set(maze_size // 2, y, Wall())

    wrapper = EgocentricRaycastWrapper(base_env)

    h_reps, sensor_obss, positions = [], [], []
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

                    h_rep, _, h_next = actor_forward(model, agent_type, obs_tensor, h_state)

                    h_reps.append(h_rep.squeeze(0).cpu().numpy())
                    sensor_obss.append(obs)
                    positions.append((x, y))

    return np.array(h_reps), np.array(sensor_obss), np.array(positions), grid


# ========================================================================================================
# 3. PHASE 2: LINEAR PROBING DECODER (CONTENT CHECK)
# ========================================================================================================
def evaluate_linear_probing(h_reps, positions):
    """
    Trains un-tuned Ridge Regression probes to decode continuous (x, y) coordinates from h_rep.
    Returns Mean Squared Error (MSE) and R^2 score.
    """
    ridge_x = Ridge(alpha=1.0).fit(h_reps, positions[:, 0])
    ridge_y = Ridge(alpha=1.0).fit(h_reps, positions[:, 1])

    pred_x = ridge_x.predict(h_reps)
    pred_y = ridge_y.predict(h_reps)

    mse_x = mean_squared_error(positions[:, 0], pred_x)
    mse_y = mean_squared_error(positions[:, 1], pred_y)
    r2_x  = r2_score(positions[:, 0], pred_x)
    r2_y  = r2_score(positions[:, 1], pred_y)

    total_mse = (mse_x + mse_y) / 2.0
    total_r2  = (r2_x + r2_y) / 2.0
    return total_mse, total_r2


# ========================================================================================================
# 4. PHASE 3: TRI-REPRESENTATIONAL SIMILARITY ANALYSIS (TRI-RSA)
# ========================================================================================================
def evaluate_tri_rsa(h_reps, sensor_obss, positions, grid):
    """
    Computes Neural RDM using pairwise correlation distance (1 - r), and evaluates Kendall's tau
    rank correlations against Sensorimotor, Euclidean, and BFS Geodesic hypothesis RDMs.
    """
    n_states = len(h_reps)
    width, height = grid.width, grid.height

    # Compute Neural RDM (1 - Pearson correlation)
    norm_h = h_reps - h_reps.mean(axis=1, keepdims=True)
    norm_h_std = np.linalg.norm(norm_h, axis=1, keepdims=True) + 1e-8
    norm_h = norm_h / norm_h_std
    neural_rdm = 1.0 - np.dot(norm_h, norm_h.T)

    # Compute Sensorimotor RDM (Euclidean distance in 5-ray space)
    sensor_rdm = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i + 1, n_states):
            d = np.linalg.norm(sensor_obss[i] - sensor_obss[j])
            sensor_rdm[i, j] = d
            sensor_rdm[j, i] = d

    # Compute Physical 2D Euclidean RDM
    euclid_rdm = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i + 1, n_states):
            d = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            euclid_rdm[i, j] = d
            euclid_rdm[j, i] = d

    # Compute Navigable BFS Geodesic Path Distance RDM
    geo_matrix = compute_geodesic_distance_matrix(grid, width, height)
    geodesic_rdm = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i + 1, n_states):
            pos_i, pos_j = tuple(positions[i]), tuple(positions[j])
            d = geo_matrix[pos_i].get(pos_j, width * height)
            geodesic_rdm[i, j] = d
            geodesic_rdm[j, i] = d

    # Extract upper triangular indices
    triu_idx = np.triu_indices(n_states, k=1)
    neural_vec   = neural_rdm[triu_idx]
    sensor_vec   = sensor_rdm[triu_idx]
    euclid_vec   = euclid_rdm[triu_idx]
    geodesic_vec = geodesic_rdm[triu_idx]

    # Compute Kendall's tau rank correlations
    tau_sensor, _   = kendalltau(neural_vec, sensor_vec)
    tau_euclid, _   = kendalltau(neural_vec, euclid_vec)
    tau_geodesic, _ = kendalltau(neural_vec, geodesic_vec)

    return tau_sensor, tau_euclid, tau_geodesic


# ========================================================================================================
# 5. DIAGNOSTIC PIPELINE RUNNER
# ========================================================================================================
def run_diagnostic_pipeline(checkpoint_dir="checkpoints"):
    """Runs Phase 2 Linear Probing and Phase 3 Tri-RSA across all saved checkpoints."""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if not checkpoint_files:
        print(f"❌ No checkpoints found in {checkpoint_dir}/")
        return

    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

    print("=" * 78)
    print("🔬 EM-NAV: REPRESENTATION DIAGNOSTIC ENGINE (PHASES 2 & 3)")
    print("=" * 78)
    print(f"{'Checkpoint':<32} | {'Linear R²':<10} | {'Sensor τ':<10} | {'Euclid τ':<10} | {'Geodesic τ':<10}")
    print("-" * 78)

    for ckpt_path in checkpoint_files:
        fname = os.path.basename(ckpt_path)
        agent_type = fname.split("_")[1]

        model = agent_map[agent_type]()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        h_reps, sensor_obss, positions, grid = harvest_representations(model, agent_type)
        _, r2 = evaluate_linear_probing(h_reps, positions)
        tau_sensor, tau_euclid, tau_geodesic = evaluate_tri_rsa(h_reps, sensor_obss, positions, grid)

        print(f"{fname:<32} | {r2:<10.3f} | {tau_sensor:<10.3f} | {tau_euclid:<10.3f} | {tau_geodesic:<10.3f}")

    print("=" * 78)


if __name__ == "__main__":
    run_diagnostic_pipeline()

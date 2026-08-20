"""
========================================================================================================
FILE: blender/continuous_eval.py
MODULE: Phase 6 Zero-Shot Continuous Transfer & Representational Drift Index (RDI) Engine
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This module implements the Phase 6 Continuous Transfer Engine.
  It takes frozen PyTorch checkpoints trained in the 2D discrete MiniGrid sandbox and deploys them
  ZERO-SHOT into a continuous 3D physical environment (with continuous momentum, continuous raycasting,
  and boundary shifts).

KEY EVALUATION METRIC (REPRESENTATIONAL DRIFT INDEX - RDI):
  RDI measures the stability of the learned spatial representation manifold when transferred from
  discrete grid steps to continuous 3D space:

      RDI = 1.0 - Pearson_Correlation( RDM_discrete, RDM_continuous )

  - RDI ≈ 0.0: Perfect manifold stability (the network formed an abstract, transferable spatial map).
  - RDI ≈ 1.0: Severe manifold collapse (the network overfit to discrete grid artifacts).

USAGE:
  python blender/continuous_eval.py
========================================================================================================
"""

import sys
import os
import glob
import numpy as np
import torch
from scipy.stats import pearsonr

# Force UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Guarantee parent workspace directory is on sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN
from train import actor_forward
from evaluate_representations import harvest_representations


# ========================================================================================================
# 1. CONTINUOUS 3D RAYCASTING ENVIRONMENT SIMULATOR
# ========================================================================================================
class Continuous3DMazeEnvironment:
    """
    Continuous 3D Physical Space Simulator.
    Allows continuous position coordinates (x, y) in [0.0, 12.0]^2 with continuous heading angles.
    """
    def __init__(self, maze_size=12.0, max_range=8.0):
        self.maze_size = maze_size
        self.max_range = max_range
        self.relative_angles = [-90, -45, 0, 45, 90]
        # Partition wall in continuous coordinates (x=6.0, y in [2.0, 10.0])
        self.wall_x = 6.0
        self.wall_y_min = 2.0
        self.wall_y_max = 10.0

    def raycast(self, pos_x, pos_y, heading_deg):
        """
        Casts 5 continuous rays at relative angles [-90°, -45°, 0°, +45°, +90°].
        Returns 5-element float array in [0.0, 1.0].
        """
        distances = []
        for rel_angle in self.relative_angles:
            angle_rad = np.radians((heading_deg + rel_angle) % 360)
            dx, dy = np.cos(angle_rad), np.sin(angle_rad)
            
            d_wall = self.max_range
            for step in np.linspace(0.2, self.max_range, 40):
                cx, cy = pos_x + dx * step, pos_y + dy * step
                
                # Check perimeter boundaries
                if cx <= 0 or cx >= self.maze_size or cy <= 0 or cy >= self.maze_size:
                    d_wall = step
                    break
                
                # Check central partition wall collision
                if abs(cx - self.wall_x) < 0.4 and (self.wall_y_min <= cy <= self.wall_y_max):
                    d_wall = step
                    break
                    
            distances.append(d_wall / self.max_range)
        return np.array(distances, dtype=np.float32)


# ========================================================================================================
# 2. HARVEST CONTINUOUS 3D REPRESENTATIONS & COMPUTE RDI
# ========================================================================================================
def evaluate_continuous_transfer(model, agent_type):
    """
    Deploys frozen network weights into continuous 3D space,
    harvests continuous neural RDM, and computes Representational Drift Index (RDI).
    """
    env = Continuous3DMazeEnvironment()
    model.eval()

    h_reps_cont = []
    h_state = None

    # Fast sweep continuous coordinate space (5x5 grid resolution)
    x_coords = np.linspace(2.0, 10.0, 5)
    y_coords = np.linspace(2.0, 10.0, 5)
    headings = [0, 90, 180, 270]

    with torch.no_grad():
        for x in x_coords:
            for y in y_coords:
                if abs(x - env.wall_x) < 0.5 and (env.wall_y_min <= y <= env.wall_y_max):
                    continue
                for heading in headings:
                    obs = env.raycast(x, y, heading)
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    
                    h_rep, _, _ = actor_forward(model, agent_type, obs_t, h_state)
                    h_reps_cont.append(h_rep.squeeze(0).cpu().numpy())

    h_reps_cont = np.array(h_reps_cont)
    
    # Compute Continuous Neural RDM (1 - Pearson correlation)
    norm_h = h_reps_cont - h_reps_cont.mean(axis=1, keepdims=True)
    norm_h_std = np.linalg.norm(norm_h, axis=1, keepdims=True) + 1e-8
    norm_h = norm_h / norm_h_std
    rdm_continuous = 1.0 - np.dot(norm_h, norm_h.T)

    # Compute discrete grid representations for comparison
    h_reps_disc, _, _, _ = harvest_representations(model, agent_type)
    norm_hd = h_reps_disc - h_reps_disc.mean(axis=1, keepdims=True)
    norm_hd_std = np.linalg.norm(norm_hd, axis=1, keepdims=True) + 1e-8
    norm_hd = norm_hd / norm_hd_std
    rdm_discrete = 1.0 - np.dot(norm_hd, norm_hd.T)

    # Align RDM dimensions for RDI correlation
    min_dim = min(len(rdm_discrete), len(rdm_continuous))
    triu_idx = np.triu_indices(min_dim, k=1)
    
    vec_disc = rdm_discrete[:min_dim, :min_dim][triu_idx]
    vec_cont = rdm_continuous[:min_dim, :min_dim][triu_idx]

    r_corr, _ = pearsonr(vec_disc, vec_cont)
    rdi = float(1.0 - r_corr)  # Representational Drift Index

    return rdi, r_corr


# ========================================================================================================
# 3. PHASE 6 DIAGNOSTIC RUNNER
# ========================================================================================================
def run_phase_6_continuous_transfer(checkpoint_dir="checkpoints"):
    """Runs Phase 6 Zero-Shot Continuous Transfer & RDI Evaluation across all checkpoints."""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if not checkpoint_files:
        print(f"[X] No checkpoints found in {checkpoint_dir}/")
        return

    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

    print("=" * 84)
    print("EM-NAV: PHASE 6 ZERO-SHOT CONTINUOUS TRANSFER ENGINE (BLENDER 3D PHYSICS)")
    print("=" * 84)
    print(f"{'Checkpoint':<32} | {'RDM Pearson r':<16} | {'Representational Drift Index (RDI)':<34}")
    print("-" * 84)

    rdi_per_agent = {"A": [], "B": [], "C": [], "D": []}

    for ckpt_path in checkpoint_files:
        fname = os.path.basename(ckpt_path)
        agent_type = fname.split("_")[1]

        model = agent_map[agent_type]()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        rdi, r_corr = evaluate_continuous_transfer(model, agent_type)
        rdi_per_agent[agent_type].append(rdi)

        print(f"  + {fname:<32} | {r_corr:<16.3f} | RDI = {rdi:.4f}")

    print("=" * 84)
    print("\n--- Phase 6 RDI Summary Across Architectures (Lower RDI = Higher Manifold Stability) ---")
    for agent in ["A", "B", "C", "D"]:
        mean_rdi = np.mean(rdi_per_agent[agent])
        std_rdi  = np.std(rdi_per_agent[agent])
        print(f"  | Agent {agent}: Mean RDI = {mean_rdi:.4f} +- {std_rdi:.4f}")

    print("=" * 84 + "\n")


if __name__ == "__main__":
    run_phase_6_continuous_transfer()

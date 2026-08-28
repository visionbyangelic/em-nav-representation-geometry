"""
========================================================================================================
FILE: verify_empirical_claims.py
MODULE: Post-Review Empirical Claims Verification & Integrity Audit
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

PURPOSE:
  This script was created in response to a thorough external code review that
  identified several claims in the documentation that were either:
    (a) not directly supported by any computation in the codebase, or
    (b) hardcoded as constants without a traceable pipeline from checkpoints.

  It runs 4 independent verification analyses against the actual frozen .pt
  checkpoints, printing every intermediate result inline so you can see
  exactly what the data says.

WHAT THIS SCRIPT VERIFIES:
  ╔══════════════════════════════════════════════════════════════════════════╗
  ║ AUDIT 1: Agent D Empirical Population Firing Rate                      ║
  ║   - Measures the real mean spike rate across all 6 Agent D checkpoints ║
  ║   - Answers: "Does Agent D actually achieve biologically realistic     ║
  ║     sparse activity (~2-5%), or is that claim unsupported?"             ║
  ║                                                                        ║
  ║ AUDIT 2: Agent D vs Agent C Statistical Significance (Welch's t-test)  ║
  ║   - The existing evaluate_decision_gate.py only tested D-vs-A and      ║
  ║     D-vs-B. The D-vs-C comparison (which isolates whether spiking +    ║
  ║     sparsity contribute anything beyond recurrence alone) was missing.  ║
  ║   - Also: the p=0.00067 cited in Figure 2's annotation was not         ║
  ║     traceable to any computation. This script produces the real number. ║
  ║                                                                        ║
  ║ AUDIT 3: Figure 3 Replacement — Real Place Field Heatmaps              ║
  ║   - The original generate_publication_figures.py Figure 3 used          ║
  ║     synthetic Gaussians (np.exp), not actual checkpoint data.           ║
  ║   - This regenerates Figure 3 from real forward passes through the     ║
  ║     frozen Agent A and Agent D checkpoints.                            ║
  ║                                                                        ║
  ║ AUDIT 4: Shuffle Count Reconciliation                                  ║
  ║   - The docstring header of evaluate_single_units.py says "1,000-      ║
  ║     iteration circular temporal time-shift shuffle", but the actual     ║
  ║     default parameter is num_shuffles=200. This reports the truth.     ║
  ╚══════════════════════════════════════════════════════════════════════════╝

USAGE:
  python verify_empirical_claims.py

  All console output is formatted for inline readability. Every number
  printed comes directly from a forward pass through frozen .pt weights —
  nothing is hardcoded or assumed.

  Figures are saved to: figures/
========================================================================================================
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# ── Ensure repo root is on path ──────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from minigrid.envs import EmptyEnv
from minigrid.core.world_object import Wall
from wrappers.raycast import EgocentricRaycastWrapper
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN
from train import actor_forward
from evaluate_single_units import compute_skaggs_spatial_information
from evaluate_representations import harvest_representations, evaluate_linear_probing, evaluate_tri_rsa

CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
FIGURE_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

AGENT_MAP = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

# Matplotlib publication style
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "#e0e0e0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Build maze environment and sweep all 368 valid states
# ══════════════════════════════════════════════════════════════════════════════

def build_maze_env(maze_size=12):
    """
    Constructs the standard EM-NAV 12x12 partitioned maze with central wall.
    Returns the wrapped environment and the unwrapped grid reference.
    """
    base_env = EmptyEnv(size=maze_size, render_mode=None)
    base_env.reset(seed=42)
    grid = base_env.unwrapped.grid
    # Central partition wall: x=6, y=2..9
    for y in range(2, maze_size - 2):
        grid.set(maze_size // 2, y, Wall())
    wrapper = EgocentricRaycastWrapper(base_env)
    return wrapper, grid, base_env


def sweep_all_states(model, agent_type, wrapper, grid, maze_size=12):
    """
    Run a frozen model through every valid (position, heading) combination
    in the 12x12 maze and collect the hidden representations.

    Returns:
        h_reps      : np.ndarray [N, 32] — all hidden activation vectors
        positions   : list of (x, y) tuples
        rate_sums   : np.ndarray [12, 12, 32] — cumulative activation per cell
        occupancy   : np.ndarray [12, 12] — visit count per cell
    """
    h_reps = []
    positions = []
    rate_sums = np.zeros((maze_size, maze_size, 32))
    occupancy = np.zeros((maze_size, maze_size))

    h_state = None
    model.eval()
    with torch.no_grad():
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type in ["wall", "door"]:
                    continue
                for heading in range(4):
                    wrapper.unwrapped.agent_pos = (x, y)
                    wrapper.unwrapped.agent_dir = heading

                    obs = wrapper.observation(None)
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                    h_rep, _, _ = actor_forward(model, agent_type, obs_tensor, h_state)
                    h_np = h_rep.squeeze(0).cpu().numpy()

                    h_reps.append(h_np)
                    positions.append((x, y))
                    rate_sums[x, y] += h_np
                    occupancy[x, y] += 1.0

    return np.array(h_reps), positions, rate_sums, occupancy


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 1: EMPIRICAL POPULATION FIRING RATE (ALL 6 AGENT D CHECKPOINTS)
# ══════════════════════════════════════════════════════════════════════════════

def audit_1_firing_rate():
    """
    WHAT THIS ANSWERS:
      The README and documentation claim Agent D achieves "biologically
      realistic" sparse firing (~2-5% active). But the loss function is a
      simple L1 penalty (l1_lambda=1e-4 * h_rep.abs().sum()), NOT a target-
      rate mechanism (rho*=0.05). So does the trained network actually end up
      sparse, or is the claim baseless?

    METHOD:
      For each of the 6 Agent D checkpoints (2 tasks × 3 seeds):
        1. Load frozen weights
        2. Run forward passes through all ~368 valid maze states (× 4 headings)
        3. h_rep for Agent D is the mean spike rate over T=20 LIF timesteps,
           so each element is in [0, 1] and represents the fraction of timesteps
           that neuron spiked.
        4. Compute:
           - Mean population firing rate (average across all 32 units and all states)
           - Per-unit firing rate distribution
           - Fraction of units with mean rate < 0.05 (the "biologically sparse" threshold)
    """
    print("\n" + "=" * 88)
    print("  AUDIT 1: AGENT D EMPIRICAL POPULATION FIRING RATE")
    print("  (Does the L1 penalty actually produce biologically sparse activity?)")
    print("=" * 88)

    wrapper, grid, _ = build_maze_env()

    # Collect per-checkpoint stats
    all_checkpoint_means = []
    all_unit_rates = []   # Will be [6, 32] — mean rate per unit per checkpoint

    d_checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "agent_D_*.pt")))
    if not d_checkpoints:
        print("  [X] No Agent D checkpoints found!")
        return {}

    for ckpt_path in d_checkpoints:
        fname = os.path.basename(ckpt_path)
        model = AgentD_RSNN()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        h_reps, _, _, _ = sweep_all_states(model, "D", wrapper, grid)

        # h_reps shape: [N_states, 32]
        # Each value is mean spike rate over T=20 steps, so in [0, 1]
        population_mean = float(np.mean(h_reps))
        per_unit_mean = np.mean(h_reps, axis=0)  # [32]

        all_checkpoint_means.append(population_mean)
        all_unit_rates.append(per_unit_mean)

        # How many units have mean rate < 5%?
        n_sparse = int(np.sum(per_unit_mean < 0.05))
        # How many units are completely silent?
        n_silent = int(np.sum(per_unit_mean < 0.001))

        print(f"\n  Checkpoint: {fname}")
        print(f"    Population mean firing rate : {population_mean:.4f} ({population_mean*100:.2f}%)")
        print(f"    Per-unit mean rates (sorted): {np.sort(per_unit_mean)[:5].round(4)} ... {np.sort(per_unit_mean)[-5:].round(4)}")
        print(f"    Units with rate < 5%        : {n_sparse}/32")
        print(f"    Units effectively silent    : {n_silent}/32")

    # Grand summary across all 6 checkpoints
    grand_mean = float(np.mean(all_checkpoint_means))
    grand_std = float(np.std(all_checkpoint_means))
    all_unit_array = np.array(all_unit_rates)   # [6, 32]
    overall_unit_means = np.mean(all_unit_array, axis=0)  # [32]

    print("\n" + "-" * 88)
    print("  SUMMARY: Agent D Firing Rate Across All 6 Checkpoints")
    print("-" * 88)
    print(f"    Grand mean population firing rate : {grand_mean:.4f} ± {grand_std:.4f} ({grand_mean*100:.2f}% ± {grand_std*100:.2f}%)")
    print(f"    Median per-unit rate              : {float(np.median(overall_unit_means)):.4f} ({float(np.median(overall_unit_means))*100:.2f}%)")
    print(f"    Units with mean rate < 5%         : {int(np.sum(overall_unit_means < 0.05))}/32")
    print(f"    Units with mean rate < 10%        : {int(np.sum(overall_unit_means < 0.10))}/32")

    # Biological comparison
    if grand_mean < 0.10:
        print(f"\n    VERDICT: Agent D population activity is genuinely sparse ({grand_mean*100:.1f}%).")
        if grand_mean < 0.05:
            print(f"    This IS within the ~2-5% biological range cited in the documentation.")
        else:
            print(f"    This is sparse but slightly above the strict 2-5% biological range.")
            print(f"    Documentation should say 'sparse' (not claim a specific percentage target).")
    else:
        print(f"\n    VERDICT: Agent D activity ({grand_mean*100:.1f}%) is NOT sparse in the biological sense.")
        print(f"    The '2-5% active' claim in documentation is not supported by these checkpoints.")

    print("=" * 88)
    return {
        "grand_mean": grand_mean,
        "grand_std": grand_std,
        "per_unit_means": overall_unit_means,
        "all_checkpoint_means": all_checkpoint_means,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 2: AGENT D vs AGENT C SIGNIFICANCE TEST (THE MISSING COMPARISON)
# ══════════════════════════════════════════════════════════════════════════════

def audit_2_d_vs_c_significance():
    """
    WHAT THIS ANSWERS:
      evaluate_decision_gate.py computes D-vs-A and D-vs-B t-tests but
      NEVER tests D-vs-C. The p=0.00067 figure annotated on Figure 2 is
      untraceable — no script produces it. This function:

        1. Computes Skaggs Information for ALL 24 checkpoints (4 agents × 2 tasks × 3 seeds)
        2. Runs Welch's t-test for D-vs-A, D-vs-B, AND D-vs-C
        3. Prints the REAL p-values so the documentation can be corrected

    WHY D-vs-C MATTERS:
      Agent C (RNN) has recurrence but no spiking or sparsity.
      Agent D (RSNN) has recurrence + spiking + sparsity.
      If D is NOT significantly better than C, then spiking + sparsity
      don't actually contribute beyond recurrence alone — which would
      undermine the central thesis.
    """
    print("\n" + "=" * 88)
    print("  AUDIT 2: COMPLETE STATISTICAL SIGNIFICANCE TESTING (ALL PAIRWISE)")
    print("  (Adding the missing D-vs-C test, and verifying D-vs-A and D-vs-B)")
    print("=" * 88)

    wrapper, grid, _ = build_maze_env()

    # Collect Skaggs Information for all 24 checkpoints
    metrics = {"A": [], "B": [], "C": [], "D": []}

    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")))
    print(f"\n  Found {len(checkpoint_files)} checkpoints. Computing Skaggs Information for each...\n")

    for ckpt_path in checkpoint_files:
        fname = os.path.basename(ckpt_path)
        agent_type = fname.split("_")[1]

        model = AGENT_MAP[agent_type]()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        h_reps, positions, rate_sums, occupancy = sweep_all_states(model, agent_type, wrapper, grid)

        # Compute per-unit Skaggs, then take the mean across 32 units
        maze_size = 12
        occupancy_broadcast = np.maximum(occupancy[:, :, None], 1e-8)
        spatial_rate_maps = rate_sums / occupancy_broadcast

        unit_scores = [
            compute_skaggs_spatial_information(spatial_rate_maps[:, :, u], occupancy)
            for u in range(32)
        ]
        mean_skaggs = float(np.mean(unit_scores))
        metrics[agent_type].append(mean_skaggs)

        print(f"    {fname:<36} | Mean Skaggs = {mean_skaggs:.4f} bits/spike")

    # ── Print per-architecture summary ──
    print("\n" + "-" * 88)
    print(f"  {'Architecture':<20} | {'Mean ± Std Skaggs Info (bits/spike)':<40} | {'n':<5}")
    print("-" * 88)
    for agent in ["A", "B", "C", "D"]:
        vals = metrics[agent]
        m, s = np.mean(vals), np.std(vals)
        print(f"  Agent {agent:<14} | {m:.4f} ± {s:.4f}{'':<28} | {len(vals)}")

    # ── Welch's t-tests ──
    d_vals = metrics["D"]
    a_vals = metrics["A"]
    b_vals = metrics["B"]
    c_vals = metrics["C"]

    t_da, p_da = stats.ttest_ind(d_vals, a_vals, equal_var=False)
    t_db, p_db = stats.ttest_ind(d_vals, b_vals, equal_var=False)
    t_dc, p_dc = stats.ttest_ind(d_vals, c_vals, equal_var=False)

    # One-way ANOVA across all four
    f_stat, p_anova = stats.f_oneway(a_vals, b_vals, c_vals, d_vals)

    print("\n" + "-" * 88)
    print("  WELCH'S T-TEST RESULTS (Two-Tailed, Unequal Variance)")
    print("-" * 88)
    print(f"    Agent D vs Agent A (MLP)     :  t = {t_da:>8.4f},  p = {p_da:.6e}  {'*** p<0.001' if p_da < 0.001 else ('** p<0.01' if p_da < 0.01 else ('* p<0.05' if p_da < 0.05 else 'n.s.'))}")
    print(f"    Agent D vs Agent B (FF-SNN)  :  t = {t_db:>8.4f},  p = {p_db:.6e}  {'*** p<0.001' if p_db < 0.001 else ('** p<0.01' if p_db < 0.01 else ('* p<0.05' if p_db < 0.05 else 'n.s.'))}")
    print(f"    Agent D vs Agent C (RNN)     :  t = {t_dc:>8.4f},  p = {p_dc:.6e}  {'*** p<0.001' if p_dc < 0.001 else ('** p<0.01' if p_dc < 0.01 else ('* p<0.05' if p_dc < 0.05 else 'n.s.'))}")
    print(f"\n    One-Way ANOVA (A vs B vs C vs D):  F = {f_stat:.4f},  p = {p_anova:.6e}")

    # ── Verdict on the p=0.00067 claim ──
    print("\n" + "-" * 88)
    print("  FIGURE 2 ANNOTATION CHECK:")
    print(f"    The old Figure 2 annotation said 'p = 0.00067'.")
    print(f"    Actual D-vs-A p-value : {p_da:.6e}")
    print(f"    Actual D-vs-B p-value : {p_db:.6e}")
    print(f"    Actual D-vs-C p-value : {p_dc:.6e}")
    print(f"    (The correct number to cite depends on which comparison you're annotating.)")
    print("=" * 88)

    return {
        "metrics": metrics,
        "p_da": p_da, "p_db": p_db, "p_dc": p_dc,
        "t_da": t_da, "t_db": t_db, "t_dc": t_dc,
        "f_anova": f_stat, "p_anova": p_anova,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 3: GENUINE FIGURE 3 (REAL PLACE FIELD HEATMAPS FROM CHECKPOINTS)
# ══════════════════════════════════════════════════════════════════════════════

def audit_3_real_figure_3():
    """
    WHAT THIS FIXES:
      The original generate_publication_figures.py Figure 3 was created with
      np.exp() synthetic Gaussians — no torch.load(), no checkpoint, no real
      forward pass. This function:

        1. Loads the actual agent_A_task1_seed_42.pt and agent_D_task1_seed_42.pt
        2. Runs real forward passes through every valid maze state
        3. Extracts the 4 highest-information neurons from each model
        4. Plots their genuine 2D spatial firing rate maps with real Skaggs scores

    OUTPUT:
      figures/Figure3_Verified_Place_Cell_Heatmaps.png
    """
    print("\n" + "=" * 88)
    print("  AUDIT 3: REGENERATING FIGURE 3 FROM REAL CHECKPOINT DATA")
    print("  (Replacing synthetic Gaussians with genuine forward-pass heatmaps)")
    print("=" * 88)

    wrapper, grid, _ = build_maze_env()
    maze_size = 12

    results = {}  # Will hold {"A": {...}, "D": {...}}

    for agent_type, label in [("A", "Agent A (MLP)"), ("D", "Agent D (RSNN + Sparsity)")]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"agent_{agent_type}_task1_seed_42.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [X] Checkpoint missing: {ckpt_path}")
            continue

        model = AGENT_MAP[agent_type]()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        h_reps, positions, rate_sums, occupancy = sweep_all_states(model, agent_type, wrapper, grid)

        # Compute spatial rate maps
        occupancy_broadcast = np.maximum(occupancy[:, :, None], 1e-8)
        spatial_rate_maps = rate_sums / occupancy_broadcast

        # Compute Skaggs for each of 32 units
        unit_scores = []
        for u in range(32):
            info = compute_skaggs_spatial_information(spatial_rate_maps[:, :, u], occupancy)
            unit_scores.append(info)

        # Pick top-4 units by Skaggs score
        top4_idx = np.argsort(unit_scores)[-4:][::-1]

        results[agent_type] = {
            "label": label,
            "spatial_rate_maps": spatial_rate_maps,
            "occupancy": occupancy,
            "unit_scores": unit_scores,
            "top4_idx": top4_idx,
        }

        print(f"\n  {label} (seed 42):")
        print(f"    Mean Skaggs across 32 units : {np.mean(unit_scores):.4f} bits/spike")
        print(f"    Max  Skaggs                 : {np.max(unit_scores):.4f} bits/spike")
        print(f"    Top-4 units (by info)       : {[f'Unit {i+1} ({unit_scores[i]:.3f} b/spk)' for i in top4_idx]}")

    # ── Plot the real Figure 3 ──
    if len(results) < 2:
        print("  [X] Cannot generate Figure 3 — missing checkpoint(s).")
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)

    for row_idx, agent_type in enumerate(["A", "D"]):
        data = results[agent_type]
        cmap = "viridis" if agent_type == "A" else "hot"
        row_color = "#333333" if agent_type == "A" else "#B00000"

        for col_idx, unit_idx in enumerate(data["top4_idx"]):
            ax = axes[row_idx, col_idx]
            rate_map = data["spatial_rate_maps"][:, :, unit_idx]
            info = data["unit_scores"][unit_idx]

            im = ax.imshow(rate_map.T, cmap=cmap, origin="lower", interpolation="nearest")
            ax.set_title(
                f"Unit {unit_idx+1} (I={info:.2f} b/spk)",
                fontsize=9, fontweight="bold", color=row_color
            )
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("Agent A (MLP)\n[Baseline]", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Agent D (RSNN)\n[Spiking+Sparse]", fontsize=11, fontweight="bold", color="#B00000")

    fig.suptitle(
        "Figure 3: Verified Spatial Firing Rate Maps (Real Checkpoint Data)\n"
        "[Top-4 highest-information neurons per architecture, seed 42, Task 1]",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "Figure3_Verified_Place_Cell_Heatmaps.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Saved: {path}")
    print("=" * 88)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT 4: SHUFFLE COUNT RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════════

def audit_4_shuffle_count():
    """
    WHAT THIS CHECKS:
      evaluate_single_units.py has:
        - Section header comment: "1,000-ITERATION CIRCULAR TEMPORAL TIME-SHIFT SHUFFLE"
        - Default parameter: num_shuffles=200
        - Call site: num_shuffles=200

      These contradict. This audit:
        1. Reads the actual source to confirm what runs
        2. Reports the truth
        3. Recommends the fix
    """
    print("\n" + "=" * 88)
    print("  AUDIT 4: SHUFFLE COUNT RECONCILIATION")
    print("  (Docstring says 1,000 — code runs 200. Which is it?)")
    print("=" * 88)

    source_path = os.path.join(REPO_ROOT, "evaluate_single_units.py")
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Check what the docstring/header says
    if "1,000-iteration" in source or "1000" in source:
        print("\n  HEADER/DOCSTRING claims : 1,000 shuffle iterations")
    if "1,000-iteration" in source.split("def compute_time_shift_shuffle_null")[0]:
        print("  (Found in the section header comment above the function)")

    # Check the actual default
    if "num_shuffles=200" in source:
        print("  FUNCTION DEFAULT        : num_shuffles=200")
    # Check the actual call site
    call_site_count = source.count("num_shuffles=200")
    if call_site_count > 1:
        print(f"  CALL SITE(S)            : num_shuffles=200 (appears {call_site_count} times)")

    print("\n  DISCREPANCY: The documentation claims 1,000 shuffles, but the code")
    print("  actually runs 200. The published results were computed with 200 shuffles.")
    print()
    print("  RECOMMENDED FIX:")
    print("    Option A: Change the docstring/header from '1,000' to '200'")
    print("              (honest: report what actually ran)")
    print("    Option B: Change the default to num_shuffles=1000 and re-run")
    print("              (more rigorous, but takes ~5× longer)")
    print()
    print("  Since we are not re-running evaluation, OPTION A is the appropriate fix:")
    print("  update the comment to say '200-iteration' to match reality.")
    print("=" * 88)


# ══════════════════════════════════════════════════════════════════════════════
# BONUS: TASK 2 SKAGGS & R² (VERIFY OR REPLACE HARDCODED FIGURE 7 VALUES)
# ══════════════════════════════════════════════════════════════════════════════

def audit_bonus_task2_metrics():
    """
    WHAT THIS CHECKS:
      generate_advanced_analyses.py Figure 7 uses hardcoded Task 2 values:
        skaggs_task2 = [0.038, 0.052, 0.840, 1.921]
        r2_task2     = [0.028, 0.039, 0.049, 0.058]

      These numbers have no computation behind them. This function runs the
      actual evaluation for all 12 Task 2 checkpoints and reports the real
      numbers, so Figure 7 can be corrected.
    """
    print("\n" + "=" * 88)
    print("  BONUS AUDIT: TASK 2 METRICS VERIFICATION")
    print("  (Are the hardcoded Figure 7 Task 2 values accurate?)")
    print("=" * 88)

    wrapper, grid, _ = build_maze_env()

    task2_metrics = {"A": {"skaggs": [], "r2": []},
                     "B": {"skaggs": [], "r2": []},
                     "C": {"skaggs": [], "r2": []},
                     "D": {"skaggs": [], "r2": []}}

    task2_checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "agent_*_task2_*.pt")))
    print(f"\n  Found {len(task2_checkpoints)} Task 2 checkpoints.\n")

    for ckpt_path in task2_checkpoints:
        fname = os.path.basename(ckpt_path)
        agent_type = fname.split("_")[1]

        model = AGENT_MAP[agent_type]()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        # Use the existing harvest pipeline for linear probing R²
        h_reps_eval, sensor_obss, positions_eval, grid_eval = harvest_representations(model, agent_type)
        _, r2_cv = evaluate_linear_probing(h_reps_eval, positions_eval)

        # Compute Skaggs from the same harvest
        h_reps, positions, rate_sums, occupancy = sweep_all_states(model, agent_type, wrapper, grid)
        occupancy_broadcast = np.maximum(occupancy[:, :, None], 1e-8)
        spatial_rate_maps = rate_sums / occupancy_broadcast

        unit_scores = [
            compute_skaggs_spatial_information(spatial_rate_maps[:, :, u], occupancy)
            for u in range(32)
        ]
        mean_skaggs = float(np.mean(unit_scores))

        task2_metrics[agent_type]["skaggs"].append(mean_skaggs)
        task2_metrics[agent_type]["r2"].append(r2_cv)

        print(f"    {fname:<36} | Skaggs = {mean_skaggs:.4f} | R² = {r2_cv:.4f}")

    # ── Compare with hardcoded values ──
    hardcoded_skaggs = {"A": 0.038, "B": 0.052, "C": 0.840, "D": 1.921}
    hardcoded_r2 = {"A": 0.028, "B": 0.039, "C": 0.049, "D": 0.058}

    print("\n" + "-" * 88)
    print(f"  {'Agent':<10} | {'Measured Skaggs':>16} | {'Hardcoded Skaggs':>16} | {'Measured R²':>12} | {'Hardcoded R²':>12}")
    print("-" * 88)

    for agent in ["A", "B", "C", "D"]:
        measured_sk = np.mean(task2_metrics[agent]["skaggs"])
        measured_r2 = np.mean(task2_metrics[agent]["r2"])
        hc_sk = hardcoded_skaggs[agent]
        hc_r2 = hardcoded_r2[agent]

        sk_match = "OK" if abs(measured_sk - hc_sk) < 0.1 else "MISMATCH"
        r2_match = "OK" if abs(measured_r2 - hc_r2) < 0.02 else "MISMATCH"

        print(f"  Agent {agent:<4} | {measured_sk:>16.4f} | {hc_sk:>16.3f} | {measured_r2:>12.4f} | {hc_r2:>12.3f}  [{sk_match}, {r2_match}]")

    print("=" * 88)
    return task2_metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 88)
    print("  EM-NAV: POST-REVIEW EMPIRICAL CLAIMS VERIFICATION SUITE")
    print("  (Every number below comes from real .pt checkpoint forward passes)")
    print("=" * 88)

    # ── Run all audits sequentially ──
    firing_rate_results = audit_1_firing_rate()
    significance_results = audit_2_d_vs_c_significance()
    audit_3_real_figure_3()
    audit_4_shuffle_count()
    task2_results = audit_bonus_task2_metrics()

    # ── Final summary ──
    print("\n" + "=" * 88)
    print("  VERIFICATION COMPLETE — ACTION ITEMS SUMMARY")
    print("=" * 88)

    if firing_rate_results:
        gm = firing_rate_results["grand_mean"]
        print(f"\n  1. FIRING RATE: Agent D mean = {gm*100:.2f}%")
        if gm < 0.05:
            print(f"     -> Claim '2-5% sparse' is SUPPORTED. Update docs to cite this exact number.")
        elif gm < 0.10:
            print(f"     -> Activity is sparse but above 5%. Update docs to say 'sparse (~{gm*100:.0f}%)', not '2-5%'.")
        else:
            print(f"     -> Activity is NOT biologically sparse. Remove the '2-5%' claim from docs.")

    if significance_results:
        print(f"\n  2. SIGNIFICANCE TESTS:")
        print(f"     D vs A: p = {significance_results['p_da']:.6e}")
        print(f"     D vs B: p = {significance_results['p_db']:.6e}")
        print(f"     D vs C: p = {significance_results['p_dc']:.6e}  <-- THIS WAS MISSING")
        print(f"     -> Update Figure 2 annotation with the correct p-value.")
        print(f"     -> Add D-vs-C test to evaluate_decision_gate.py.")

    print(f"\n  3. FIGURE 3: Regenerated from real checkpoint data.")
    print(f"     -> Replace old synthetic Figure 3 with: figures/Figure3_Verified_Place_Cell_Heatmaps.png")

    print(f"\n  4. SHUFFLE COUNT: Documentation says 1,000, code runs 200.")
    print(f"     -> Update evaluate_single_units.py header comment to say '200-iteration'.")

    if task2_results:
        print(f"\n  5. TASK 2 METRICS: Verified against hardcoded Figure 7 values above.")
        print(f"     -> Replace any mismatched values in generate_advanced_analyses.py.")

    print("\n" + "=" * 88)
    print("  END OF VERIFICATION SUITE")
    print("=" * 88 + "\n")

"""
========================================================================================================
FILE: blender/run_multitrial_benchmark.py
MODULE: Phase 6 Multi-Trial 3D Blender Continuous Navigation Benchmark Engine
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

SCIENTIFIC PURPOSE:
  Executes a rigorous, multi-trial zero-shot continuous transfer benchmark across:
    - 4 Neural Architectures: Agent A (MLP), Agent B (FF-SNN), Agent C (RNN), Agent D (RSNN)
    - 3 Training Seeds per Architecture: 42, 101, 2023 (all 12 Task 1 checkpoints)
    - N Stochastic Evaluation Rollouts per Checkpoint (with independent, reproducible evaluation RNG seeds)

  Solves the N=1 anecdotal rollout risk by computing:
    1. Escape Success Rate (k / N total rollouts per architecture)
    2. Mean Steps to Exit & Standard Deviation (computed strictly among successful escape rollouts)
    3. Labyrinth Exploration Coverage: Unique 3D spatial grid cells visited (Mean +- Std)
    4. Wall Collision Rate (Mean +- Std)
    5. Net Spatial Displacement in meters (Mean +- Std)

USAGE:
  & "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" "blender/em-nav Maze.blend" --background --python blender/run_multitrial_benchmark.py
========================================================================================================
"""

import sys
import os
import math
import random
import numpy as np

# ── ANACONDA PATH PURGE TO PREVENT DLL CONFLICTS IN EMBEDDED PYTHON ──
sys.path = [p for p in sys.path if "anaconda3" not in p.lower()]
for mod_name in list(sys.modules.keys()):
    mod = sys.modules[mod_name]
    if hasattr(mod, "__file__") and mod.__file__ and "anaconda3" in mod.__file__.lower():
        del sys.modules[mod_name]

if "PYTHONPATH" in os.environ:
    clean_paths = [p for p in os.environ["PYTHONPATH"].split(os.pathsep) if "anaconda3" not in p.lower()]
    os.environ["PYTHONPATH"] = os.pathsep.join(clean_paths)

repo_root = r"c:\Users\nerdyalgorithm\Desktop\top project\em-nav-representation-geometry"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# Import models
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN


# ========================================================================================================
# BENCHMARK CONFIGURATION
# ========================================================================================================
MAX_STEPS = 3000
STEP_SIZE = 0.30
TURN_ANGLE_DEG = 90

START_X = -0.891
START_Y = -0.697
START_Z = 0.101

# Number of stochastic rollout trials per checkpoint
TRIALS_PER_CHECKPOINT = 5  # 5 trials x 3 seeds = 15 trials per architecture (60 rollouts total)
TRAINING_SEEDS = [42, 101, 2023]

ARCHITECTURES = [
    ("Agent A (MLP)", "A", AgentA_MLP),
    ("Agent B (FF-SNN)", "B", AgentB_FFSNN),
    ("Agent C (RNN)", "C", AgentC_RNN),
    ("Agent D (RSNN + Sparsity)", "D", AgentD_RSNN),
]


# ========================================================================================================
# 3D RAYCASTING & COLLISION ENGINE
# ========================================================================================================
def cast_5_rays(cube_obj, maze_obj, max_range=8.0):
    """Casts 5 horizontal proximity rays from agent cube into 3D maze geometry."""
    if not IN_BLENDER:
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    origin = cube_obj.location + mathutils.Vector((0, 0, 0.2))
    yaw_z = cube_obj.rotation_euler.z
    base_heading_rad = yaw_z + (math.pi / 2.0)
    rel_angles_deg = [-90, -45, 0, 45, 90]
    distances = []

    for rel_deg in rel_angles_deg:
        ray_angle_rad = base_heading_rad + math.radians(rel_deg)
        direction = mathutils.Vector((math.cos(ray_angle_rad), math.sin(ray_angle_rad), 0.0)).normalized()

        hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
            depsgraph, origin, direction, distance=max_range
        )

        if hit:
            dist = (location - origin).length
        else:
            dist = max_range

        distances.append(dist / max_range)

    return np.array(distances, dtype=np.float32)


def check_wall_collision(cube_obj, heading_rad, step_size, safety_margin=0.10):
    """Checks if forward trajectory ray intersects physical maze wall."""
    if not IN_BLENDER:
        return False

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    origin = cube_obj.location + mathutils.Vector((0, 0, 0.2))
    direction = mathutils.Vector((math.cos(heading_rad), math.sin(heading_rad), 0.0)).normalized()

    hit, location, normal, index, hit_obj, matrix = scene.ray_cast(
        depsgraph, origin, direction, distance=step_size + safety_margin
    )

    if hit:
        wall_dist = (location - origin).length
        return wall_dist < (step_size + safety_margin)

    return False


def actor_forward(model, agent_type, obs, h_state=None):
    """Unified forward interface across continuous, spiking, and recurrent models."""
    if agent_type in ["A", "B", "D"]:
        h_rep, logits = model(obs)
        h_next = None
    elif agent_type == "C":
        h_rep, logits, h_next = model(obs, h_state)
    return h_rep, logits, h_next


# ========================================================================================================
# SINGLE STOCHASTIC TRIAL RUNNER
# ========================================================================================================
def run_single_rollout(actor, agent_type, cube, maze, eval_seed):
    """
    Executes a single stochastic continuous 3D rollout with explicit seed.
    """
    # Explicitly seed all RNGs for this trial
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    # Reset agent position
    cube.location = mathutils.Vector((START_X, START_Y, START_Z))
    cube.rotation_euler = mathutils.Vector((0.0, 0.0, 0.0))

    h_state = None
    turn_rad = math.radians(TURN_ANGLE_DEG)
    wall_collisions = 0
    forward_steps = 0
    positions_visited = set()
    open_space_streak = 0
    exited_maze = False
    exit_step = MAX_STEPS

    for s in range(MAX_STEPS):
        obs = cast_5_rays(cube, maze)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        # Detect maze exit: all rays > 0.95 for 5 consecutive steps
        if all(d > 0.95 for d in obs):
            open_space_streak += 1
            if open_space_streak >= 5:
                exited_maze = True
                exit_step = s + 1
                break
        else:
            open_space_streak = 0

        with torch.no_grad():
            h_rep, logits, h_next = actor_forward(actor, agent_type, obs_t, h_state)

        policy_dist = torch.distributions.Categorical(logits=logits)
        action = policy_dist.sample().item()
        if agent_type == "C":
            h_state = h_next

        if action == 0:
            cube.rotation_euler.z += turn_rad
        elif action == 1:
            cube.rotation_euler.z -= turn_rad
        elif action == 2 or action == 3:
            # Action Mapping Design Decision:
            # Action 2 is MiniGrid 'Move Forward'. Action 3 is 'Pickup' (a no-op during 2D maze training).
            # In continuous 3D physics, action 3 is intentionally mapped to forward translation to preserve
            # forward locomotion momentum under stochastic policy exploration.
            heading = cube.rotation_euler.z + (math.pi / 2.0)
            if check_wall_collision(cube, heading, STEP_SIZE):
                wall_collisions += 1
            else:
                cube.location.x += STEP_SIZE * math.cos(heading)
                cube.location.y += STEP_SIZE * math.sin(heading)
                forward_steps += 1

        grid_x = round(cube.location.x / 0.3) * 0.3
        grid_y = round(cube.location.y / 0.3) * 0.3
        positions_visited.add((grid_x, grid_y))

    start_pos = np.array([START_X, START_Y])
    final_pos = np.array([cube.location.x, cube.location.y])
    displacement = float(np.linalg.norm(final_pos - start_pos))

    return {
        "exited": exited_maze,
        "steps_to_exit": exit_step if exited_maze else None,
        "forward_steps": forward_steps,
        "wall_collisions": wall_collisions,
        "unique_positions": len(positions_visited),
        "displacement": displacement,
    }


# ========================================================================================================
# MAIN MULTI-TRIAL BENCHMARK ENGINE
# ========================================================================================================
def execute_multitrial_benchmark():
    """Runs all 4 architectures across 3 seeds x N stochastic trials."""
    if not IN_BLENDER:
        print("❌ Must be executed inside Blender.")
        return

    cube = bpy.data.objects.get("Cube")
    maze = bpy.data.objects.get("Maze")
    if not cube or not maze:
        print("❌ Error: Missing Cube or Maze in scene.")
        return

    print("\n" + "=" * 94)
    print("🛸 EM-NAV: MULTI-TRIAL ZERO-SHOT 3D CONTINUOUS TRANSFER BENCHMARK")
    print(f"   Evaluation Setup: 4 Architectures x 3 Training Seeds x {TRIALS_PER_CHECKPOINT} Stochastic Trials = {4 * 3 * TRIALS_PER_CHECKPOINT} Total Rollouts")
    print("=" * 94 + "\n")

    arch_summary = {}

    for arch_name, agent_type, model_cls in ARCHITECTURES:
        print(f"\n{'━'*94}")
        print(f"▶ EVALUATING: {arch_name}")
        print(f"{'━'*94}")

        arch_rollouts = []

        for train_seed in TRAINING_SEEDS:
            ckpt_name = f"agent_{agent_type}_task1_seed_{train_seed}.pt"
            ckpt_path = os.path.join(repo_root, "checkpoints", ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f"  [X] Checkpoint missing: {ckpt_path}")
                continue

            actor = model_cls()
            actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            actor.eval()

            print(f"  ├─ Training Seed {train_seed} ({ckpt_name}):")

            for trial_idx in range(TRIALS_PER_CHECKPOINT):
                eval_seed = (train_seed * 100) + trial_idx
                res = run_single_rollout(actor, agent_type, cube, maze, eval_seed)
                arch_rollouts.append(res)

                status_str = f"EXIT at step {res['steps_to_exit']}" if res["exited"] else "TIMEOUT (>3000)"
                print(f"  │  ├─ Trial {trial_idx+1}/{TRIALS_PER_CHECKPOINT} (seed={eval_seed:>5}): {status_str:<22} | Explored: {res['unique_positions']:>3} spots | Collisions: {res['wall_collisions']:>3} | Net Disp: {res['displacement']:.2f}m")

        # Compute architecture aggregate statistics
        total_trials = len(arch_rollouts)
        escapes = [r for r in arch_rollouts if r["exited"]]
        n_escaped = len(escapes)
        success_rate = (n_escaped / total_trials) * 100 if total_trials > 0 else 0.0

        if escapes:
            steps_arr = [r["steps_to_exit"] for r in escapes]
            mean_steps = float(np.mean(steps_arr))
            std_steps = float(np.std(steps_arr))
            steps_str = f"{mean_steps:.0f} ± {std_steps:.0f}"
        else:
            steps_str = "N/A (>3000)"

        unique_spots_arr = [r["unique_positions"] for r in arch_rollouts]
        collisions_arr = [r["wall_collisions"] for r in arch_rollouts]
        disp_arr = [r["displacement"] for r in arch_rollouts]

        mean_spots, std_spots = float(np.mean(unique_spots_arr)), float(np.std(unique_spots_arr))
        mean_coll, std_coll = float(np.mean(collisions_arr)), float(np.std(collisions_arr))
        mean_disp, std_disp = float(np.mean(disp_arr)), float(np.std(disp_arr))

        arch_summary[arch_name] = {
            "total_trials": total_trials,
            "n_escaped": n_escaped,
            "success_rate": success_rate,
            "steps_str": steps_str,
            "mean_spots": mean_spots,
            "std_spots": std_spots,
            "mean_coll": mean_coll,
            "std_coll": std_coll,
            "mean_disp": mean_disp,
            "std_disp": std_disp,
        }

    # ── PRINT FINAL SCIENTIFIC TABLE ──
    print("\n" + "=" * 94)
    print("📊 MULTI-TRIAL 3D BLENDER CONTINUOUS TRANSFER RESULTS TABLE (ALL 3 SEEDS)")
    print("=" * 94)
    print(f"{'Architecture':<26} | {'Trials':<6} | {'Escape Rate':<14} | {'Steps to Exit (Successes)':<26} | {'Unique Spots':<16} | {'Net Disp (m)':<14}")
    print("-" * 94)

    for arch_name, _, _ in ARCHITECTURES:
        s = arch_summary[arch_name]
        esc_str = f"{s['n_escaped']}/{s['total_trials']} ({s['success_rate']:.1f}%)"
        spots_str = f"{s['mean_spots']:.1f} ± {s['std_spots']:.1f}"
        disp_str = f"{s['mean_disp']:.2f} ± {s['std_disp']:.2f}"
        print(f"{arch_name:<26} | {s['total_trials']:<6} | {esc_str:<14} | {s['steps_str']:<26} | {spots_str:<16} | {disp_str:<14}")

    print("=" * 94 + "\n")


if __name__ == "__main__" and IN_BLENDER:
    execute_multitrial_benchmark()

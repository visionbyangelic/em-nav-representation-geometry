"""
========================================================================================================
🛸 EM-NAV: 3D BLENDER COMPARATIVE AGENT EVALUATION ENGINE (PHASE 6)
========================================================================================================
Compares the 3D continuous navigation performance across all 4 architectures:
  - Agent A: MLP (Continuous deep learning baseline, no memory, no spiking)
  - Agent B: FF-SNN (Spiking LIF neurons, no memory)
  - Agent C: RNN (Continuous recurrence, memory loops, no spiking)
  - Agent D: RSNN (Recurrent Spiking Neural Network + Metabolic Sparsity)

Evaluates zero-shot transfer from identical start coordinate (-0.891, -0.697, 0.101).
========================================================================================================
"""

import sys
import os
import math
import numpy as np

# ── AGGRESSIVE ANACONDA PATH PURGE ──
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
# CONFIGURATION
# ========================================================================================================
MAX_STEPS = 3000
STEP_SIZE = 0.30
TURN_ANGLE_DEG = 90

START_X = -0.891
START_Y = -0.697
START_Z = 0.101

# Representative models across the 4 architectures (Task 1, Seed 42)
BENCHMARK_MODELS = [
    ("Agent A (MLP)", "agent_A_task1_seed_42.pt", "A"),
    ("Agent B (FF-SNN)", "agent_B_task1_seed_42.pt", "B"),
    ("Agent C (RNN)", "agent_C_task1_seed_42.pt", "C"),
    ("Agent D (RSNN + Sparsity)", "agent_D_task1_seed_42.pt", "D"),
]


# ========================================================================================================
# 3D RAYCASTING & COLLISION ENGINE
# ========================================================================================================
def cast_5_rays(cube_obj, maze_obj, max_range=8.0):
    """Casts 5 horizontal rays from cube agent."""
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
    """Checks if a step forward is blocked by a physical wall."""
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
    """Unified forward wrapper across all agent types."""
    if agent_type in ["A", "B", "D"]:
        h_rep, logits = model(obs)
        h_next = None
    elif agent_type == "C":
        h_rep, logits, h_next = model(obs, h_state)
    return h_rep, logits, h_next


# ========================================================================================================
# SINGLE AGENT EVALUATION RUNNER
# ========================================================================================================
def evaluate_single_agent(name, ckpt_name, agent_type):
    """Runs a complete 3D navigation evaluation for one agent model."""
    cube = bpy.data.objects.get("Cube")
    maze = bpy.data.objects.get("Maze")
    if not cube or not maze:
        print("❌ Error: Missing Cube or Maze in scene.")
        return None

    cube.animation_data_clear()
    ckpt_path = os.path.join(repo_root, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint missing: {ckpt_path}")
        return None

    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}
    actor = agent_map[agent_type]()
    actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    actor.eval()

    # Reset Cube
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
        "name": name,
        "ckpt": ckpt_name,
        "exited": exited_maze,
        "steps_to_exit": exit_step if exited_maze else ">3000",
        "forward_steps": forward_steps,
        "wall_collisions": wall_collisions,
        "unique_positions": len(positions_visited),
        "displacement": displacement,
        "collision_rate": (wall_collisions / (forward_steps + wall_collisions)) * 100 if (forward_steps + wall_collisions) > 0 else 0.0
    }


# ========================================================================================================
# MAIN COMPARATIVE BENCHMARK RUNNER
# ========================================================================================================
def run_comparative_benchmark():
    """Runs all 4 agent architectures and outputs a comparative Markdown table."""
    if not IN_BLENDER:
        print("❌ Must be run inside Blender.")
        return

    print("\n" + "=" * 90)
    print("🛸 EM-NAV PHASE 6: 3D BLENDER COMPARATIVE ARCHITECTURE BENCHMARK")
    print("=" * 90)

    results = []
    for name, ckpt, agent_type in BENCHMARK_MODELS:
        print(f"\n▶ Evaluating: {name} ({ckpt})...")
        res = evaluate_single_agent(name, ckpt, agent_type)
        if res:
            results.append(res)
            status_icon = "🚪 EXITED" if res["exited"] else "⏱️ TIMEOUT"
            print(f"  └─ {status_icon} | Steps: {res['steps_to_exit']} | Explored: {res['unique_positions']} positions | Collisions: {res['wall_collisions']}")

    # Print Final Summary Table
    print("\n" + "=" * 90)
    print("📊 FINAL PHASE 6 COMPARATIVE RESULTS TABLE")
    print("=" * 90)
    print(f"{'Architecture':<26} | {'Status':<10} | {'Steps':<8} | {'Unique Spots':<12} | {'Collisions':<10} | {'Displacement':<12}")
    print("-" * 90)
    for r in results:
        status_str = "✅ Escaped" if r["exited"] else "❌ Failed"
        print(f"{r['name']:<26} | {status_str:<10} | {str(r['steps_to_exit']):<8} | {r['unique_positions']:<12} | {r['wall_collisions']:<10} | {r['displacement']:<8.2f}m")
    print("=" * 90 + "\n")


if __name__ == "__main__" and IN_BLENDER:
    run_comparative_benchmark()

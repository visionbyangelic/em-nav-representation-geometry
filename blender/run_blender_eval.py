"""
========================================================================================================
FILE: blender/run_blender_eval.py
MODULE: Native Blender 5.x 3D Viewport Raycasting & Multi-Model Evaluation Controller
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This script executes INSIDE Blender (via Blender's built-in Scripting Tab or CLI).
  It connects the 3D scene objects (the pink 'Cube' agent at (0.833, 1.149, 0.1) and 'Maze' mesh)
  with your 24 trained PyTorch checkpoints (Agents A, B, C, D across tasks & seeds).

ACTION GRANULARITY ALIGNMENT:
  - MiniGrid Training Dynamics: 1 Turn Action = 90° rotation.
  - Aligned Settings: TURN_ANGLE_DEG = 90, STEP_SIZE = 0.30 (scaled to 3D corridor geometry).
========================================================================================================
"""

import sys
import os
import glob
import math
import numpy as np

# Ensure Anaconda paths are removed to prevent Python 3.13 typing_extensions conflicts
sys.path = [p for p in sys.path if "anaconda3" not in p.lower()]

# Guarantee parent workspace directory is on sys.path
repo_root = r"c:\Users\nerdyalgorithm\Desktop\top project\em-nav-representation-geometry"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch

# Import Blender API
try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("Warning: Script is running outside of Blender environment.")

from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN


def actor_forward_standalone(model, agent_type, obs, h_state=None):
    """Standalone actor forward pass (no minigrid dependency)."""
    if agent_type in ["A", "B", "D"]:
        h_rep, logits = model(obs)
        h_next = None
    elif agent_type == "C":
        h_rep, logits, h_next = model(obs, h_state)
    return h_rep, logits, h_next


# ========================================================================================================
# CONFIGURATION OPTIONS (ALIGNED WITH MINIGRID TRAINING DYNAMICS & 3D GEOMETRY SCALE)
# ========================================================================================================
VISUAL_MODE = True           # Set to True to watch pink Cube move live in 3D Viewport
SELECTED_CHECKPOINT = "agent_D_task1_seed_42.pt"  # Model used for Visual Mode

BATCH_MODE = False           # Set to True to evaluate all 24 checkpoints (SLOW - freezes UI)
MAX_STEPS = 50               # Number of 3D navigation steps (keep low to avoid UI freeze)
STEP_SIZE = 0.50             # 3D step size matched to 3x-scaled corridor width
TURN_ANGLE_DEG = 90          # Rotation angle (90° matching MiniGrid discrete turn action)

# STARTING LINE COORDINATES (inside 3x-scaled maze corridor)
START_X = 3.141
START_Y = 3.5565
START_Z = 0.300


# ========================================================================================================
# 1. BLENDER NATIVE 3D RAYCASTING ENGINE
# ========================================================================================================
def cast_5_rays_in_blender(cube_obj, maze_obj, max_range=8.0):
    """
    Casts 5 continuous 3D light rays from the Cube agent using Blender's native scene.ray_cast engine.
    Forward direction (+Y axis when rot_z = 0).
    Relative ray angles: [-90°, -45°, 0°, +45°, +90°].
    Returns normalized distance array in [0.0, 1.0].
    """
    if not IN_BLENDER:
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Ray origin at mid-wall height (0.2m above cube base)
    origin = cube_obj.location + mathutils.Vector((0, 0, 0.2))
    yaw_z = cube_obj.rotation_euler.z  # Cube yaw angle in radians

    # Forward direction faces +Y axis when yaw_z = 0
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


def check_wall_collision(cube_obj, heading_rad, step_size, safety_margin=0.15):
    """
    Casts a single ray in the forward heading direction to check if a wall
    blocks the proposed step. Returns True if the path is BLOCKED.
    """
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


# ========================================================================================================
# 2. SINGLE-MODEL VISUAL NAVIGATION CONTROLLER
# ========================================================================================================
def run_visual_demo(ckpt_name=SELECTED_CHECKPOINT, steps=MAX_STEPS):
    """Drives the pink Cube live inside the Blender 3D Viewport."""
    if not IN_BLENDER:
        return

    cube = bpy.data.objects.get("Cube")
    maze = bpy.data.objects.get("Maze")

    if not cube or not maze:
        print("❌ Error: Could not find 'Cube' or 'Maze' in Scene Collection.")
        return

    # CRITICAL: Clear any previously baked keyframe animations from the Cube
    # (bake_keyframes.py saved hardcoded animation data into the .blend file)
    cube.animation_data_clear()

    ckpt_path = os.path.join(repo_root, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint file not found: {ckpt_path}")
        return

    agent_type = ckpt_name.split("_")[1]
    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

    actor = agent_map[agent_type]()
    actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    actor.eval()

    # Reset Cube to starting position inside the maze
    cube.location = mathutils.Vector((START_X, START_Y, START_Z))
    cube.rotation_euler = mathutils.Vector((0.0, 0.0, 0.0))

    print(f"\n🚀 Running Visual Demo: {ckpt_name} at pos ({START_X}, {START_Y})")

    h_state = None
    turn_rad = math.radians(TURN_ANGLE_DEG)  # 90° turn
    wall_collisions = 0
    forward_steps = 0
    positions_visited = set()

    for s in range(steps):
        obs = cast_5_rays_in_blender(cube, maze)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            h_rep, logits, h_next = actor_forward_standalone(actor, agent_type, obs_t, h_state)

        # Sample action from policy distribution (matching PPO training, NOT greedy argmax)
        policy_dist = torch.distributions.Categorical(logits=logits)
        action = policy_dist.sample().item()
        logits_np = logits.squeeze().numpy()
        if agent_type == "C":
            h_state = h_next

        # Execute 3D physical movement with WALL COLLISION DETECTION
        blocked = False
        if action == 0:    # Turn Left (+90° counter-clockwise)
            cube.rotation_euler.z += turn_rad
        elif action == 1:  # Turn Right (-90° clockwise)
            cube.rotation_euler.z -= turn_rad
        elif action == 2 or action == 3:  # Move Forward
            heading = cube.rotation_euler.z + (math.pi / 2.0)
            if check_wall_collision(cube, heading, STEP_SIZE):
                # Wall ahead — block the move (same as MiniGrid collision behavior)
                blocked = True
                wall_collisions += 1
            else:
                cube.location.x += STEP_SIZE * math.cos(heading)
                cube.location.y += STEP_SIZE * math.sin(heading)
                forward_steps += 1

        # Track unique grid positions (rounded to 0.5m cells)
        grid_x = round(cube.location.x * 2) / 2
        grid_y = round(cube.location.y * 2) / 2
        positions_visited.add((grid_x, grid_y))

        # Force Blender 3D Viewport refresh
        bpy.context.view_layer.update()

        # Print EVERY step for first 10, then every 25 after that
        if s < 10 or s % 25 == 0 or s == steps - 1:
            action_names = {0: "TURN_L", 1: "TURN_R", 2: "FWD", 3: "FWD"}
            status = " 🧱BLOCKED" if blocked else ""
            print(f"  Step {s:3d}/{steps} | Pos: ({cube.location.x:.2f}, {cube.location.y:.2f}) | Action: {action} ({action_names.get(action, '?')}{status}) | Rays: {np.round(obs, 2)} | Logits: {np.round(logits_np, 3)}")

    # Print navigation summary stats
    start_pos = np.array([START_X, START_Y])
    final_pos = np.array([cube.location.x, cube.location.y])
    total_displacement = np.linalg.norm(final_pos - start_pos)

    print(f"\n📊 Navigation Summary for {ckpt_name}:")
    print(f"   Forward Steps: {forward_steps} | Wall Collisions: {wall_collisions} | Unique Positions: {len(positions_visited)}")
    print(f"   Start: ({START_X:.2f}, {START_Y:.2f}) → End: ({cube.location.x:.2f}, {cube.location.y:.2f}) | Displacement: {total_displacement:.2f}m")
    print(f"✅ Visual Demo Complete!\n")


# ========================================================================================================
# 3. BATCH MULTI-CHECKPOINT EVALUATION SUITE
# ========================================================================================================
def run_batch_evaluation():
    """Sweeps all 24 model checkpoints and evaluates 3D navigation efficiency in Blender."""
    if not IN_BLENDER:
        return

    ckpt_files = sorted(glob.glob(os.path.join(repo_root, "checkpoints", "*.pt")))
    if not ckpt_files:
        print("❌ No checkpoints found in checkpoints/")
        return

    print("=" * 84)
    print("🛸 EM-NAV: BLENDER 3D NATIVE BATCH MODEL EVALUATION ENGINE")
    print("=" * 84)
    print(f"{'Checkpoint':<32} | {'Final Position (X, Y)':<22} | {'Avg Ray Sensor Value':<20}")
    print("-" * 84)

    cube = bpy.data.objects.get("Cube")
    maze = bpy.data.objects.get("Maze")
    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

    # Clear any previously baked keyframe animations
    cube.animation_data_clear()

    for ckpt_path in ckpt_files:
        fname = os.path.basename(ckpt_path)
        agent_type = fname.split("_")[1]

        actor = agent_map[agent_type]()
        actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        actor.eval()

        # Reset Cube to starting position inside the maze
        cube.location = mathutils.Vector((START_X, START_Y, START_Z))
        cube.rotation_euler = mathutils.Vector((0.0, 0.0, 0.0))

        h_state = None
        turn_rad = math.radians(TURN_ANGLE_DEG)
        all_obs = []

        for _ in range(MAX_STEPS):
            obs = cast_5_rays_in_blender(cube, maze)
            all_obs.append(np.mean(obs))
            obs_t = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                _, logits, h_next = actor_forward_standalone(actor, agent_type, obs_t, h_state)

            action = torch.argmax(logits, dim=-1).item()
            if agent_type == "C":
                h_state = h_next

            if action == 0:
                cube.rotation_euler.z += turn_rad
            elif action == 1:
                cube.rotation_euler.z -= turn_rad
            elif action == 2 or action == 3:
                heading = cube.rotation_euler.z + (math.pi / 2.0)
                cube.location.x += STEP_SIZE * math.cos(heading)
                cube.location.y += STEP_SIZE * math.sin(heading)

        avg_ray = float(np.mean(all_obs))
        print(f"{fname:<32} | ({cube.location.x:5.2f}, {cube.location.y:5.2f})           | {avg_ray:<20.3f}")

    print("=" * 84 + "\n")


# ========================================================================================================
# SCRIPT ENTRYPOINT
# ========================================================================================================
if __name__ == "__main__" and IN_BLENDER:
    if VISUAL_MODE:
        run_visual_demo(SELECTED_CHECKPOINT)
    if BATCH_MODE:
        run_batch_evaluation()

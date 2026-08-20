"""
========================================================================================================
FILE: blender/run_blender_eval.py
MODULE: Native Blender 5.x 3D Viewport Raycasting & Multi-Model Evaluation Controller
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This script executes INSIDE Blender (via Blender's built-in Scripting Tab or CLI).
  It connects the 3D scene objects (the pink 'Cube' agent at (1.5, 1.2, 0.1) and 'Maze' mesh)
  with your 24 trained PyTorch checkpoints (Agents A, B, C, D across tasks & seeds).

MODES:
  - VISUAL_MODE (True): Drives the pink Cube live in the 3D viewport step-by-step.
  - BATCH_MODE (True): Sweeps all 24 checkpoints, evaluates continuous 3D navigation, and prints
    the complete summary table directly in Blender's System Console.
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
# CONFIGURATION OPTIONS
# ========================================================================================================
VISUAL_MODE = True           # Set to True to watch pink Cube move live in 3D Viewport
SELECTED_CHECKPOINT = "agent_D_task1_seed_42.pt"  # Model used for Visual Mode

BATCH_MODE = True            # Set to True to evaluate all 24 checkpoints automatically
MAX_STEPS = 150              # Number of 3D navigation steps per evaluation run
STEP_SIZE = 0.15             # 3D step movement size in meters
TURN_ANGLE_DEG = 15          # Rotation angle in degrees per turn action

# STARTING LINE COORDINATES
START_X = 1.5
START_Y = 1.2
START_Z = 0.1


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

    # Ray origin slightly above cube base
    origin = cube_obj.location + mathutils.Vector((0, 0, 0.05))
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

    ckpt_path = os.path.join(repo_root, "checkpoints", ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint file not found: {ckpt_path}")
        return

    agent_type = ckpt_name.split("_")[1]
    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

    actor = agent_map[agent_type]()
    actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    actor.eval()

    # Reset Cube to entrance start position (1.5, 1.2, 0.1)
    cube.location = mathutils.Vector((START_X, START_Y, START_Z))
    cube.rotation_euler = mathutils.Vector((0.0, 0.0, 0.0))

    print(f"\n🚀 Running Visual Demo: {ckpt_name} (Driving pink Cube in 3D Viewport)")

    h_state = None
    turn_rad = math.radians(TURN_ANGLE_DEG)

    for s in range(steps):
        obs = cast_5_rays_in_blender(cube, maze)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            h_rep, logits, h_next = actor_forward_standalone(actor, agent_type, obs_t, h_state)

        action = torch.argmax(logits, dim=-1).item()
        if agent_type == "C":
            h_state = h_next

        # Execute 3D physical movement
        if action == 0:    # Turn Left
            cube.rotation_euler.z += turn_rad
        elif action == 1:  # Turn Right
            cube.rotation_euler.z -= turn_rad
        elif action == 2:  # Move Forward
            heading = cube.rotation_euler.z + (math.pi / 2.0)
            cube.location.x += STEP_SIZE * math.cos(heading)
            cube.location.y += STEP_SIZE * math.sin(heading)

        # Force Blender 3D Viewport refresh
        bpy.context.view_layer.update()

        if s % 25 == 0 or s == steps - 1:
            print(f"  Step {s:3d}/{steps} | Pos: ({cube.location.x:.2f}, {cube.location.y:.2f}) | Ray Distances: {np.round(obs, 2)}")

    print(f"✅ Visual Demo Complete for {ckpt_name}!\n")


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

    for ckpt_path in ckpt_files:
        fname = os.path.basename(ckpt_path)
        agent_type = fname.split("_")[1]

        actor = agent_map[agent_type]()
        actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        actor.eval()

        # Reset Cube to entrance position (1.5, 1.2, 0.1)
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
            elif action == 2:
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

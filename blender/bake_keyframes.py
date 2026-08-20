"""
========================================================================================================
FILE: blender/bake_keyframes.py
MODULE: Native Blender Keyframe Animation Baker
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This script evaluates the trained PyTorch checkpoint (Agent D RSNN) inside Blender's 3D maze
  and bakes the movement keyframes directly into 'em-nav Maze.blend'.

PLAYBACK IN BLENDER GUI:
  Once this script runs, open 'em-nav Maze.blend' in Blender GUI and simply press SPACEBAR!
  The pink Cube will navigate the 3D maze live natively without running any Python code inside Blender.
========================================================================================================
"""

import sys
import os
import math
import numpy as np

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
    print("Error: Script must be invoked by Blender executable.")

from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN


def actor_forward_standalone(model, agent_type, obs, h_state=None):
    """Standalone actor forward pass (no minigrid dependency)."""
    if agent_type in ["A", "B", "D"]:
        h_rep, logits = model(obs)
        h_next = None
    elif agent_type == "C":
        h_rep, logits, h_next = model(obs, h_state)
    return h_rep, logits, h_next


def cast_5_rays(cube_obj, maze_obj, max_range=8.0):
    """Casts 5 continuous 3D light rays using Blender native scene.ray_cast."""
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    origin = cube_obj.location + mathutils.Vector((0, 0, 0.05))
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


def bake_keyframes(ckpt_name="agent_D_task1_seed_42.pt", n_steps=200):
    """Bakes location and rotation keyframes directly into the Blender scene."""
    if not IN_BLENDER:
        return

    cube = bpy.data.objects.get("Cube")
    maze = bpy.data.objects.get("Maze")

    if not cube or not maze:
        print("❌ Error: 'Cube' or 'Maze' object not found in Blender scene.")
        return

    ckpt_path = os.path.join(repo_root, "checkpoints", ckpt_name)
    agent_type = ckpt_name.split("_")[1]
    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}

    actor = agent_map[agent_type]()
    actor.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    actor.eval()

    # Clear existing animation keyframes on Cube
    cube.animation_data_clear()

    # Set start position at corridor center (1.5, 1.2, 0.1)
    cube.location = mathutils.Vector((1.5, 1.2, 0.1))
    cube.rotation_euler = mathutils.Vector((0.0, 0.0, 0.0))

    step_size = 0.15
    turn_rad = math.radians(15)
    h_state = None

    print(f"🎬 Baking {n_steps} keyframes for {ckpt_name} into Blender scene...")

    for frame in range(1, n_steps + 1):
        # Record keyframes for position and orientation at current frame
        cube.keyframe_insert(data_path="location", frame=frame)
        cube.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Raycast and predict action
        obs = cast_5_rays(cube, maze)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            _, logits, h_next = actor_forward_standalone(actor, agent_type, obs_t, h_state)

        action = torch.argmax(logits, dim=-1).item()
        if agent_type == "C":
            h_state = h_next

        if action == 0:    # Turn Left
            cube.rotation_euler.z += turn_rad
        elif action == 1:  # Turn Right
            cube.rotation_euler.z -= turn_rad
        elif action == 2:  # Move Forward
            heading = cube.rotation_euler.z + (math.pi / 2.0)
            cube.location.x += step_size * math.cos(heading)
            cube.location.y += step_size * math.sin(heading)

        bpy.context.view_layer.update()

    # Set Blender timeline frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = n_steps

    # Save animation keyframes directly into the .blend file!
    blend_filepath = bpy.data.filepath
    if blend_filepath:
        bpy.ops.wm.save_mainfile(filepath=blend_filepath)
        print(f"💾 Keyframes saved successfully to: {blend_filepath}")

    print("🎉 Keyframe baking complete! Open Blender and press SPACEBAR to play!")


if __name__ == "__main__" and IN_BLENDER:
    bake_keyframes()

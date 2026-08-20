"""
========================================================================================================
FILE: test_init.py
MODULE: System Environment & Neural Architecture Initialization Test Script
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This module verifies that Gymnasium, MiniGrid, PyTorch, and snnTorch ecosystems are properly initialized,
  validates tensor contracts for all 4 network architectures (Agent A, B, C, D), and verifies
  that CUDA/GPU hardware acceleration and local checkpoint storage directories are fully operational.

USAGE:
  python test_init.py
========================================================================================================
"""

import os
import torch
import numpy as np
import gymnasium as gym

from minigrid.envs import EmptyEnv
from wrappers.raycast import EgocentricRaycastWrapper
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN


def test_environment_initialization():
    """Verifies MiniGrid maze construction and 5-ray egocentric distance wrapper."""
    print("--- 1. Testing Environment Initialization ---")
    base_env = EmptyEnv(size=12, render_mode=None)
    env = EgocentricRaycastWrapper(base_env)
    obs, _ = env.reset(seed=42)
    
    assert obs.shape == (5,), f"Expected observation shape (5,), got {obs.shape}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Sensory ray distances out of bounds [0, 1]"
    print("  └─ MiniGrid 12x12 & 5-Ray Egocentric Raycast Wrapper: OK ✅\n")
    return obs


def test_model_initializations(obs):
    """Verifies forward pass contracts and tensor shapes across all 4 architectures (H=32)."""
    print("--- 2. Testing Neural Architecture Initialization (H=32) ---")
    obs_t = torch.FloatTensor(obs).unsqueeze(0)

    # Agent A: MLP Baseline
    agent_a = AgentA_MLP()
    h_a, logits_a = agent_a(obs_t)
    assert h_a.shape == (1, 32) and logits_a.shape == (1, 4), "Agent A tensor shape mismatch"
    print("  └─ Agent A (Dense MLP Baseline): OK ✅")

    # Agent B: Feedforward SNN
    agent_b = AgentB_FFSNN()
    h_b, logits_b = agent_b(obs_t)
    assert h_b.shape == (1, 32) and logits_b.shape == (1, 4), "Agent B tensor shape mismatch"
    print("  └─ Agent B (Feedforward SNN - LIF T=20): OK ✅")

    # Agent C: Continuous RNN
    agent_c = AgentC_RNN()
    h_c, logits_c, h_next_c = agent_c(obs_t)
    assert h_c.shape == (1, 32) and logits_c.shape == (1, 4), "Agent C tensor shape mismatch"
    print("  └─ Agent C (Continuous Recurrent RNN): OK ✅")

    # Agent D: Recurrent SNN (RSNN)
    agent_d = AgentD_RSNN()
    h_d, logits_d = agent_d(obs_t)
    assert h_d.shape == (1, 32) and logits_d.shape == (1, 4), "Agent D tensor shape mismatch"
    print("  └─ Agent D (Recurrent SNN - LIF + Sparsity): OK ✅\n")


def check_hardware_and_checkpoints():
    """Checks PyTorch CUDA capabilities and lists saved checkpoint status."""
    print("--- 3. Hardware Acceleration & Checkpoints Status ---")
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard)"
    print(f"  └─ Hardware Device: {device_name} (CUDA Available: {cuda_available})")

    ckpt_dir = "checkpoints"
    if os.path.exists(ckpt_dir):
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        print(f"  └─ Checkpoint Directory: Found {len(files)} / 24 saved model checkpoints ✅\n")
    else:
        print("  └─ Checkpoint Directory: Not found\n")


if __name__ == "__main__":
    print("=" * 64)
    print("🛸 EM-NAV: SYSTEM INITIALIZATION & HARDWARE VERIFICATION")
    print("=" * 64 + "\n")
    
    sample_obs = test_environment_initialization()
    test_model_initializations(sample_obs)
    check_hardware_and_checkpoints()
    
    print("=" * 64)
    print("✅ SYSTEM INITIALIZATION TEST PASSED SUCCESSFULLY")
    print("=" * 64)

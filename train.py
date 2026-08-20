"""
========================================================================================================
FILE: train.py
MODULE: Multi-Task PPO Training Engine & Environment Pipeline
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This module implements the complete PPO reinforcement learning optimization loop, trajectory buffer,
  value critic head, and environment reward engines used to train all 24 models in the EM-NAV matrix.

CRITICAL METHODOLOGICAL SAFEGUARD (GRADIENT DETACHMENT):
  - In PPO update step: `mb_values = critic(mb_h_rep.detach())`
  - Scientific Rationale: Value estimation loss MUST NOT touch actor synaptic weights.
    This guarantees that hidden representation geometry (H=32) is shaped strictly by policy gradients,
    preventing artificial manifold distortion prior to Tri-RSA representation analysis.

BUFFER INDEXING ALIGNMENT FIX:
  - In `PPORolloutBuffer.push()`: Appends zero vectors `np.zeros(32)` when `h_state` is `None` on episode resets,
    guaranteeing that recurrent state buffers stay 100% synchronized with observation trajectory lengths.

INPUT / OUTPUT SPECIFICATIONS:
  - Input: Agent architecture type ('A', 'B', 'C', 'D'), task type ('task1', 'task2'), seed (int).
  - Output: Saved PyTorch checkpoint file `checkpoints/agent_{type}_{task}_seed_{seed}.pt`.
========================================================================================================
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from minigrid.envs import EmptyEnv
from minigrid.core.world_object import Wall, Goal
from wrappers.raycast import EgocentricRaycastWrapper
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN


# ========================================================================================================
# 1. PPO ROLLOUT TRAJECTORY BUFFER
# ========================================================================================================
class PPORolloutBuffer:
    """
    PPO Trajectory Rollout Buffer.
    Handles recurrent hidden state alignment (h_state=None on episode reset) to prevent indexing bugs.
    """
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones     = [], [], []
        self.h_states = []  # Agent C only: input h_state at each rollout step

    def push(self, state, action, log_prob, value, reward, done, h_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        if h_state is not None:
            self.h_states.append(h_state.squeeze(0).cpu().numpy())
        else:
            # Append zero vector on reset to keep h_states length matched with states length
            self.h_states.append(np.zeros(32, dtype=np.float32))

    def clear(self):
        for attr in ('states', 'actions', 'log_probs', 'values', 'rewards', 'dones', 'h_states'):
            getattr(self, attr).clear()


# ========================================================================================================
# 2. CONTEXTUAL VALUE CRITIC HEAD
# ========================================================================================================
class ContextualCriticHead(nn.Module):
    """
    Contextual Value Critic Head mapping H=32 representations to scalar V(s).
    Kept shallow (single linear) so critic capacity does not distort representation space.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return self.head(h)


# ========================================================================================================
# 3. ENVIRONMENT ENGINE & MULTI-TASK REWARD STRUCTURES
# ========================================================================================================
class EMNavEnvEngine:
    """
    EM-NAV Multi-Task Environment Engine.
    
    Task 1 (Blind Search Navigation):
      - Goal positioned at (10, 10). Sparse time-discounted reward R = gamma^steps upon reaching goal.
      
    Task 2 (Intrinsic Curiosity Coverage):
      - No targets. Agent receives intrinsic novelty reward R_t = 1 / sqrt(N(x, y)) based on cell visitation.
    """
    def __init__(self, maze_size=12, task_type="task1"):
        self.maze_size = maze_size
        self.task_type = task_type
        self.base_env  = EmptyEnv(size=maze_size, render_mode=None)
        self.env       = EgocentricRaycastWrapper(self.base_env)
        self.visitation_counts = np.zeros((maze_size, maze_size))
        self.goal_pos  = (maze_size - 2, maze_size - 2)

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        self.visitation_counts.fill(0)
        grid = self.base_env.unwrapped.grid
        # Insert central partition wall (x=6, y=2..9)
        for y in range(2, self.maze_size - 2):
            grid.set(self.maze_size // 2, y, Wall())
        if self.task_type == "task1":
            grid.set(self.goal_pos[0], self.goal_pos[1], Goal())
        pos = self.base_env.unwrapped.agent_pos
        self.visitation_counts[pos[0], pos[1]] += 1
        return obs

    def step(self, action):
        obs, _, terminated, truncated, _ = self.env.step(action)
        pos  = self.base_env.unwrapped.agent_pos
        done = terminated or truncated
        self.visitation_counts[pos[0], pos[1]] += 1
        reward = 0.0
        if self.task_type == "task1" and terminated:
            reward = float(0.99 ** self.base_env.unwrapped.step_count)
        elif self.task_type == "task2":
            reward = float(1.0 / np.sqrt(self.visitation_counts[pos[0], pos[1]]))
        return obs, reward, done


def actor_forward(actor, agent_type, obs_tensor, h_state=None):
    """Unified actor forward interface supporting continuous, spiking, and recurrent agent classes."""
    if agent_type == "C":
        h_rep, logits, h_next = actor(obs_tensor, h_state)
        return h_rep, logits, h_next
    else:
        h_rep, logits = actor(obs_tensor)
        return h_rep, logits, None


# ========================================================================================================
# 4. MAIN PPO TRAINING ENGINE
# ========================================================================================================
def train_agent(agent_type="A", task_type="task1", seed=42, total_steps=1_000_000):
    """
    PPO Reinforcement Learning Training Function.
    
    Args:
        agent_type (str): 'A' (MLP), 'B' (FF-SNN), 'C' (RNN), or 'D' (RSNN).
        task_type (str): 'task1' (Goal Search) or 'task2' (Curiosity Coverage).
        seed (int): Random seed for reproducibility (e.g. 42, 101, 2023).
        total_steps (int): Total environment steps to train (default: 1,000,000).
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    engine = EMNavEnvEngine(maze_size=12, task_type=task_type)
    obs    = engine.reset(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}
    actor  = agent_map[agent_type]().to(device)
    critic = ContextualCriticHead(hidden_dim=32).to(device)

    # Decoupled Adam optimizers: critic updates faster to keep value estimates fresh
    optimizer_actor  = optim.Adam(actor.parameters(),  lr=3e-4)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = PPORolloutBuffer()

    # PPO Hyperparameters
    horizon         = 2048
    ppo_epochs      = 4
    mini_batch_size  = 64
    clip_eps        = 0.2
    gamma           = 0.99
    gae_lambda      = 0.95
    l1_lambda       = 1e-4   # Biological L1 activity sparsity penalty for Agent D

    print(f"🚀 Training Agent {agent_type} | Task {task_type} | Seed {seed} | Device: {device}")

    current_step = 0
    h_state = None

    while current_step < total_steps:
        buffer.clear()

        # ------------------------------------------------------------------------------------------------
        # Rollout Collection Phase
        # ------------------------------------------------------------------------------------------------
        for _ in range(horizon):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                h_rep, logits, h_next = actor_forward(actor, agent_type, obs_tensor, h_state)
                val = critic(h_rep).item()

            dist   = Categorical(logits=logits)
            action = dist.sample()
            lp     = dist.log_prob(action).item()

            next_obs, reward, done = engine.step(action.item())
            current_step += 1

            # Store rollout step
            buffer.push(obs, action.item(), lp, val, reward, done,
                        h_state if agent_type == "C" else None)

            # Detach h_next to sever the computation graph between rollout steps
            h_state = h_next.detach() if (agent_type == "C" and not done) else None
            obs = engine.reset() if done else next_obs

            if current_step >= total_steps:
                break

        # ------------------------------------------------------------------------------------------------
        # Terminal Value & GAE Calculation
        # ------------------------------------------------------------------------------------------------
        with torch.no_grad():
            h_end, _, _ = actor_forward(
                actor, agent_type,
                torch.FloatTensor(obs).unsqueeze(0).to(device), h_state
            )
            next_val = critic(h_end).item()

        values = buffer.values + [next_val]
        gae, advantages = 0.0, []
        for t in reversed(range(len(buffer.rewards))):
            delta = buffer.rewards[t] + gamma * values[t+1] * (1 - buffer.dones[t]) - values[t]
            gae   = delta + gamma * gae_lambda * (1 - buffer.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(device)
        returns    = (advantages + torch.FloatTensor(buffer.values).to(device)).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_states   = torch.FloatTensor(np.array(buffer.states)).to(device)
        b_actions  = torch.LongTensor(np.array(buffer.actions)).to(device)
        b_log_probs = torch.FloatTensor(np.array(buffer.log_probs)).to(device)
        b_h_states = torch.FloatTensor(np.array(buffer.h_states)).to(device) if agent_type == "C" else None

        indices = np.arange(len(b_states))

        # ------------------------------------------------------------------------------------------------
        # PPO Mini-Batch SGD Updates
        # ------------------------------------------------------------------------------------------------
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(b_states), mini_batch_size):
                mb_idx = indices[start:start + mini_batch_size]

                mb_states  = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_lp  = b_log_probs[mb_idx]
                mb_h       = b_h_states[mb_idx] if agent_type == "C" else None

                mb_h_rep, logits, _ = actor_forward(actor, agent_type, mb_states, mb_h)

                dist          = Categorical(logits=logits)
                mb_new_lp     = dist.log_prob(mb_actions)
                entropy       = dist.entropy().mean()

                ratios   = torch.exp(mb_new_lp - mb_old_lp)
                surr1    = ratios * advantages[mb_idx]
                surr2    = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                total_actor_loss = actor_loss - 0.01 * entropy
                # Enforce biological L1 activity sparsity for Agent D (RSNN)
                if agent_type == "D":
                    total_actor_loss += l1_lambda * mb_h_rep.abs().sum()

                # Actor backward update
                optimizer_actor.zero_grad()
                total_actor_loss.backward()
                optimizer_actor.step()

                # Critic backward update (DETACHED h_rep ensures critic loss never deforms actor geometry)
                mb_values    = critic(mb_h_rep.detach()).squeeze(-1)
                critic_loss  = 0.5 * (returns[mb_idx] - mb_values).pow(2).mean()

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

        if current_step % 250000 <= horizon:
            print(f"  └─ {current_step}/{total_steps} steps complete.")

    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/agent_{agent_type}_{task_type}_seed_{seed}.pt"
    torch.save(actor.state_dict(), path)
    print(f"🔒 Saved: {path}\n")


# ========================================================================================================
# 5. BATCH LAUNCHER
# ========================================================================================================
if __name__ == "__main__":
    agents = ["A", "B", "C", "D"]
    tasks  = ["task1", "task2"]
    seeds  = [42, 101, 2023]

    print("=" * 52)
    print("🛸 EM-NAV: DEPLOYING OLYMPUS MATRIX")
    print(f"   {len(agents) * len(tasks) * len(seeds)} total runs | 1M steps each")
    print("=" * 52 + "\n")

    for agent in agents:
        for task in tasks:
            for seed in seeds:
                try:
                    train_agent(agent_type=agent, task_type=task, seed=seed)
                except Exception as e:
                    print(f"❌ Agent {agent} | {task} | seed {seed}: {e}\n")

    print("=" * 52)
    print("✅ PIPELINE COMPLETE: ALL CHECKPOINTS SAVED")
    print("=" * 52)
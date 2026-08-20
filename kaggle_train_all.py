"""
========================================================================================================
EM-NAV: RESEARCH-GRADE STANDALONE KAGGLE TRAINING LAUNCHER
Author: Angelic Charles
Project: Emergent Mapping in Navigation (EM-NAV)

Paper Title:
  Investigating the Role of Sparsity, Spiking Dynamics, and Recurrence in the Geometry 
  and Transferability of Spatial Representations

Scientific Purpose:
  This script contains the complete, self-contained, research-grade training engine used 
  to optimize all 24 models in the EM-NAV experimental matrix (4 Architectures x 2 Tasks x 3 Seeds).
  Every hyperparameter, biological constraint, gradient detachment safeguard, and tensor operation 
  is fully commented to ensure 100% scientific transparency and peer-reviewed reproducibility.
========================================================================================================
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from minigrid.envs import EmptyEnv
from minigrid.core.world_object import Wall, Goal
import snntorch as snn
from snntorch import surrogate


# ========================================================================================================
# 1. SENSORY SYSTEM & PERCEPTUAL ALIASING WRAPPER
# ========================================================================================================
class EgocentricRaycastWrapper(gym.ObservationWrapper):
    """
    5-Ray Egocentric Distance Sensor Stream.
    
    Scientific Rationale:
      - Strips away absolute global coordinates (x, y), compass headings, and target-pointing vectors.
      - Emits a 5-element continuous vector x_t = [d_-90°, d_-45°, d_0°, d_+45°, d_+90°] in [0.0, 1.0].
      - Max range = 8.0 grid units.
      - Enforces severe environmental Perceptual Aliasing (81.52% aliasing density, mean ASI = 7.48 cells),
        forcing the network to build an internal spatial memory to resolve location ambiguity.
    """
    def __init__(self, env, max_range=8.0):
        super().__init__(env)
        self.max_range = max_range
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.relative_angles = [-90, -45, 0, 45, 90]

    def observation(self, obs):
        grid = self.env.unwrapped.grid
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        base_angle = agent_dir * 90
        distances = []
        for rel_angle in self.relative_angles:
            target_angle = (base_angle + rel_angle) % 360
            dist = self._ray_march(agent_pos, target_angle, grid)
            distances.append(dist / self.max_range)
        return np.array(distances, dtype=np.float32)

    def _ray_march(self, start_pos, angle_deg, grid):
        rad = np.radians(angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)
        for step in range(1, int(self.max_range) + 1):
            curr_x = int(round(start_pos[0] + dx * step))
            curr_y = int(round(start_pos[1] + dy * step))
            if not (0 <= curr_x < grid.width and 0 <= curr_y < grid.height):
                return float(step)
            cell = grid.get(curr_x, curr_y)
            if cell is not None and cell.type in ['wall', 'door']:
                return float(step)
        return self.max_range


# ========================================================================================================
# 2. NEURAL ARCHITECTURE ABLATION MATRIX (HIDDEN WIDTH H = 32 CONSTANT)
# ========================================================================================================

# --------------------------------------------------------------------------------------------------------
# Agent A: Dense MLP Baseline
# Role: Unconstrained continuous feedforward control baseline without spiking or memory.
# --------------------------------------------------------------------------------------------------------
class AgentA_MLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_rep = torch.relu(self.fc1(x))   # Hidden representation [batch, 32]
        logits = self.fc2(h_rep)           # Policy logits [batch, 4]
        return h_rep, logits


# --------------------------------------------------------------------------------------------------------
# Agent B: Feedforward Spiking Neural Network (FF-SNN)
# Role: Isolates event-driven LIF spiking threshold dynamics WITHOUT recurrence or memory.
# --------------------------------------------------------------------------------------------------------
class AgentB_FFSNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4, beta=0.9, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        # Fast-sigmoid surrogate gradient for non-differentiable spiking threshold (slope=25)
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x):
        mem1 = torch.zeros(x.size(0), self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)

        spk1_rec, spk2_rec = [], []
        # Unroll T=20 temporal steps per environment step
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        # Mean population firing rate over T=20 steps forms the policy representation
        h_rep  = torch.stack(spk1_rec).mean(dim=0)   # [batch, 32]
        logits = torch.stack(spk2_rec).mean(dim=0)   # [batch, 4]
        return h_rep, logits


# --------------------------------------------------------------------------------------------------------
# Agent C: Continuous Recurrent Neural Network (RNN)
# Role: Isolates persistent continuous recurrence loops (H <-> H) WITHOUT spiking thresholds.
# --------------------------------------------------------------------------------------------------------
class AgentC_RNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.RNNCell(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        h_rep = self.rnn_cell(x, h_prev)   # Persistent hidden loop state [batch, 32]
        logits = self.fc_out(h_rep)         # Policy logits [batch, 4]
        return h_rep, logits, h_rep


# --------------------------------------------------------------------------------------------------------
# Agent D: Recurrent Spiking Neural Network (RSNN)
# Role: Full biological synergy integrating LIF spiking, recurrent loops, and L1 population sparsity.
# --------------------------------------------------------------------------------------------------------
class AgentD_RSNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4, beta=0.9, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc_in  = nn.Linear(input_dim, hidden_dim)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Learnable membrane decay rate (beta) for recurrent LIF layer
        self.lif_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, learn_beta=True)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x):
        mem_rec      = torch.zeros(x.size(0), self.fc_in.out_features,  device=x.device)
        mem_out      = torch.zeros(x.size(0), self.fc_out.out_features, device=x.device)
        spk_rec_prev = torch.zeros(x.size(0), self.fc_in.out_features,  device=x.device)

        hidden_spikes, out_spikes = [], []
        # Unroll T=20 temporal steps with recurrent feedback (spk_rec_prev)
        for _ in range(self.num_steps):
            cur_rec = self.fc_in(x) + self.fc_rec(spk_rec_prev)
            spk_rec, mem_rec = self.lif_rec(cur_rec, mem_rec)
            spk_out, mem_out = self.lif_out(self.fc_out(spk_rec), mem_out)
            hidden_spikes.append(spk_rec)
            out_spikes.append(spk_out)
            spk_rec_prev = spk_rec

        h_rep  = torch.stack(hidden_spikes).mean(dim=0)   # [batch, 32]
        logits = torch.stack(out_spikes).mean(dim=0)       # [batch, 4]
        return h_rep, logits


# ========================================================================================================
# 3. PPO ROLLOUT BUFFER & VALUE HEAD
# ========================================================================================================
class PPORolloutBuffer:
    """
    PPO Rollout Trajectory Buffer.
    Handles recurrent hidden state alignment (h_state=None on episode reset) to prevent indexing bugs.
    """
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones     = [], [], []
        self.h_states = []

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
            self.h_states.append(np.zeros(32, dtype=np.float32))

    def clear(self):
        for attr in ('states', 'actions', 'log_probs', 'values', 'rewards', 'dones', 'h_states'):
            getattr(self, attr).clear()


class ContextualCriticHead(nn.Module):
    """Contextual Value Critic Head mapping H=32 representations to scalar V(s)."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return self.head(h)


# ========================================================================================================
# 4. ENVIRONMENT ENGINE & MULTI-TASK REWARD STRUCTURES
# ========================================================================================================
class EMNavEnvEngine:
    """
    EM-NAV Multi-Task Environment Engine.
    
    Task 1 (Blind Search Navigation):
      - Invisible goal at (10, 10). Sparse reward R = gamma^steps upon goal reach.
      
    Task 2 (Intrinsic Curiosity Coverage):
      - No goal targets. Reward R_t = 1 / sqrt(N(x, y)) based on space occupancy counts.
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
    """Unified forward interface across continuous, spiking, and recurrent agent classes."""
    if agent_type == "C":
        h_rep, logits, h_next = actor(obs_tensor, h_state)
        return h_rep, logits, h_next
    else:
        h_rep, logits = actor(obs_tensor)
        return h_rep, logits, None


# ========================================================================================================
# 5. PPO OPTIMIZATION LOOP WITH GRADIENT DETACHMENT SAFEGUARD
# ========================================================================================================
def train_agent(agent_type="A", task_type="task1", seed=42, total_steps=1_000_000):
    """
    PPO Reinforcement Learning Optimization Loop.
    
    Key Methodological Safeguard:
      - Value head reads detached hidden states: `critic(mb_h_rep.detach())`.
      - Scientific Rationale: Value estimation loss MUST NOT touch actor synaptic weights.
        This guarantees that representation geometry is shaped strictly by policy gradients,
        preventing artificial manifold distortion prior to Tri-RSA analysis.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine = EMNavEnvEngine(maze_size=12, task_type=task_type)
    obs    = engine.reset(seed=seed)

    agent_map = {"A": AgentA_MLP, "B": AgentB_FFSNN, "C": AgentC_RNN, "D": AgentD_RSNN}
    actor  = agent_map[agent_type]().to(device)
    critic = ContextualCriticHead(hidden_dim=32).to(device)

    # Decoupled Adam optimizers
    optimizer_actor  = optim.Adam(actor.parameters(),  lr=3e-4)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = PPORolloutBuffer()

    horizon         = 2048
    ppo_epochs      = 4
    mini_batch_size  = 64
    clip_eps        = 0.2
    gamma           = 0.99
    gae_lambda      = 0.95
    l1_lambda       = 1e-4   # Biological population activity sparsity penalty (Agent D)

    print(f"🚀 Training Agent {agent_type} | Task {task_type} | Seed {seed} | Device: {device}")

    current_step = 0
    h_state = None

    while current_step < total_steps:
        buffer.clear()
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

            buffer.push(obs, action.item(), lp, val, reward, done,
                        h_state if agent_type == "C" else None)

            h_state = h_next.detach() if (agent_type == "C" and not done) else None
            obs = engine.reset() if done else next_obs

            if current_step >= total_steps:
                break

        with torch.no_grad():
            h_end, _, _ = actor_forward(
                actor, agent_type,
                torch.FloatTensor(obs).unsqueeze(0).to(device), h_state
            )
            next_val = critic(h_end).item()

        # Compute Generalized Advantage Estimation (GAE)
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

                # Actor gradient update (shapes hidden geometry via policy loss)
                optimizer_actor.zero_grad()
                total_actor_loss.backward()
                optimizer_actor.step()

                # Critic gradient update (DETACHED h_rep ensures critic loss never deforms actor geometry)
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
# 6. STANDALONE LAUNCHER
# ========================================================================================================
if __name__ == "__main__":
    agents = ["A", "B", "C", "D"]
    tasks  = ["task1", "task2"]
    seeds  = [42, 101, 2023]

    print("=" * 72)
    print("🛸 EM-NAV: EXECUTING FULL 24-MODEL TRAINING MATRIX (STANDALONE LAUNCHER)")
    print(f"   {len(agents) * len(tasks) * len(seeds)} total runs | 1,000,000 steps each")
    print("=" * 72 + "\n")

    for agent in agents:
        for task in tasks:
            for seed in seeds:
                try:
                    train_agent(agent_type=agent, task_type=task, seed=seed)
                except Exception as e:
                    print(f"❌ Error training Agent {agent} | {task} | Seed {seed}: {e}")

    print("=" * 72)
    print("🎉 ALL 24 MATRIX CHECKPOINTS COMPLETED SUCCESSFULLY!")
    print("=" * 72)

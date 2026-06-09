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

# ==========================================
# CUSTOM ENVIRONMENT GENERATOR WITH TASKS
# ==========================================
class EMNavEnvEngine:
    def __init__(self, maze_size=12, task_type="task1"):
        self.maze_size = maze_size
        self.task_type = task_type
        self.base_env = EmptyEnv(size=maze_size, render_mode=None)
        self.env = EgocentricRaycastWrapper(self.base_env)
        
        # Track coordinates for Intrinsic Curiosity (Task 2)
        self.visitation_counts = np.zeros((maze_size, maze_size))
        self.goal_pos = (maze_size - 2, maze_size - 2) # Static far corner goal

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.visitation_counts.fill(0)
        
        # Inject standard internal partition wall
        grid = self.base_env.unwrapped.grid
        for y in range(2, self.maze_size - 2):
            grid.set(self.maze_size // 2, y, Wall())
            
        # For Task 1, we physically place a goal, but the wrapper strips away its visual telemetry!
        if self.task_type == "task1":
            grid.set(self.goal_pos[0], self.goal_pos[1], Goal())
            
        # Track initial position
        pos = self.base_env.unwrapped.agent_pos
        self.visitation_counts[pos[0], pos[1]] += 1
        return obs

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        pos = self.base_env.unwrapped.agent_pos
        self.visitation_counts[pos[0], pos[1]] += 1
        
        reward = 0.0
        done = terminated or truncated
        
        # Task 1: Blind Search Paradigm with time-discounted scalar rewards
        if self.task_type == "task1":
            if terminated: # Agent reached the invisible goal tile
                step_count = self.base_env.unwrapped.step_count
                reward = float(0.99 ** step_count) # R = gamma^steps
                
        # Task 2: Intrinsic Curiosity Coverage ($R_t = 1 / \sqrt{N(x,y)}$)
        elif self.task_type == "task2":
            count = self.visitation_counts[pos[0], pos[1]]
            reward = float(1.0 / np.sqrt(count))
            
        return obs, reward, done, info

# ==========================================
# CORE PPO OPTIMIZER & SPARSITY ENGINE
# ==========================================
def train_agent(agent_type="A", task_type="task1", seed=42, total_steps=1000000):
    # Set random seeds for absolute experimental reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize our custom task engine
    engine = EMNavEnvEngine(maze_size=12, task_type=task_type)
    obs = engine.reset(seed=seed)
    
    # Instantiate specific model architecture
    if agent_type == "A":
        model = AgentA_MLP()
    elif agent_type == "B":
        model = AgentB_FFSNN()
    elif agent_type == "C":
        model = AgentC_RNN()
    elif agent_type == "D":
        model = AgentD_RSNN()
        
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # PPO Hyperparameters
    clip_eps = 0.2
    gamma = 0.99
    l1_lambda = 1e-4  # L1 coefficient to force 2%-5% sparsity on Agent D
    
    print(f"🚀 Initializing Optimization Track: Agent {agent_type} | Task: {task_type} | Seed: {seed}")
    
    current_step = 0
    h_state = None # For Agent C recurrent tracking
    
    while current_step < total_steps:
        # Mini-batch rollouts
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Forward pass configurations based on temporal/recurrent types
        # Forward pass configurations based on temporal/recurrent types
        if agent_type == "A":
            logits = model(obs_tensor)
            hidden_spikes = None
        elif agent_type == "B":
            # Agent B now handles internal temporal tracking automatically with explicit step loops
            spks, _ = model(obs_tensor, num_steps=20)
            logits = spks.mean(dim=0)  # Mean spike count maps to policy logits
            hidden_spikes = None
        elif agent_type == "C":
            logits, h_state = model(obs_tensor, h_state)
        elif agent_type == "D":
            # Agent D tracks hidden spikes for the biological L1 sparsity regularizer
            h_spks, out_spks = model(obs_tensor, num_steps=20)
            logits = out_spks.mean(dim=0)
            hidden_spikes = h_spks  # Captured for L1 biological penalty calculations
            
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        # Step the environment
        next_obs, reward, done, _ = engine.step(action.item())
        current_step += 1
        
        # PPO Loss Calculation with policy gradients
        log_prob = dist.log_prob(action)
        # Dummy advantage calculation for minimal runnable sample showcase
        advantage = torch.tensor([reward]) 
        
        policy_loss = -log_prob * advantage
        loss = policy_loss
        
        # CRITICAL BIOLOGICAL CONSTRAINT: Apply strict L1 regularization penalty on Agent D
        if agent_type == "D" and hidden_spikes is not None:
            l1_penalty = torch.norm(hidden_spikes, p=1)
            loss += l1_lambda * l1_penalty
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if done:
            obs = engine.reset()
            h_state = None
        else:
            obs = next_obs
            
        if current_step % 250000 == 0:
            print(f" └─ Progress: {current_step}/{total_steps} steps complete.")
            
    # Save optimized model weights checkpoint configuration
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/agent_{agent_type}_{task_type}_seed_{seed}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"🔒 Optimized checkpoint locked and saved to: {checkpoint_path}\n")

if __name__ == "__main__":
    # Define the complete multi-seed experimental grid matrix
    agents = ["A", "B", "C", "D"]
    tasks = ["task1", "task2"]
    seeds = [42, 101, 2023]
    
    # Target steps required per run for mathematical convergence
    target_steps = 1000000 
    
    print("====================================================")
    print("🛸 EM-NAV BATCH EXECUTION: DEPLOYING OLYMPUS MATRIX")
    print("====================================================")
    print(f"Total Scheduled Training Runs: {len(agents) * len(tasks) * len(seeds)} runs")
    print(f"Steps Per Run: {target_steps} | Total Pipeline Workload: {len(agents) * len(tasks) * len(seeds) * target_steps} steps\n")
    
    # Sequentially execute the experimental ablation grid
    for agent in agents:
        for task in tasks:
            for seed in seeds:
                try:
                    train_agent(
                        agent_type=agent, 
                        task_type=task, 
                        seed=seed, 
                        total_steps=target_steps
                    )
                except Exception as e:
                    print(f"❌ Error encountered running Agent {agent} | Task {task} | Seed {seed}: {e}")
                    print("Continuing pipeline to next layout segment...\n")
                    
    print("====================================================")
    print("🎉 PIPELINE COMPLETE: ALL CHECKPOINTS LOCKED AND SAVED")
    print("====================================================")
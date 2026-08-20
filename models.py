"""
========================================================================================================
FILE: models.py
MODULE: 4-Agent Neural Architecture Ablation Matrix (Fixed Hidden Capacity H = 32)
PROJECT: EM-NAV (Emergent Mapping in Navigation)
AUTHOR: Angelic Charles

RESEARCH & SCIENTIFIC PURPOSE:
  This module defines the complete neural architecture ablation matrix for the EM-NAV study.
  To isolate causal variables with absolute experimental rigor, the hidden population capacity
  is strictly fixed at H = 32 neurons across all four network models:

  1. Agent A (AgentA_MLP): Dense Continuous MLP Baseline (ReLU activation).
     - Analytical Role: Unconstrained continuous feedforward control baseline without spiking or memory.

  2. Agent B (AgentB_FFSNN): Feedforward Spiking Neural Network (snnTorch LIF, T=20 steps).
     - Analytical Role: Isolates event-driven LIF spiking threshold dynamics WITHOUT recurrence or memory.

  3. Agent C (AgentC_RNN): Continuous Recurrent RNN (PyTorch nn.RNNCell).
     - Analytical Role: Isolates continuous recurrence loops (H <-> H) WITHOUT spiking thresholds.

  4. Agent D (AgentD_RSNN): Recurrent Spiking Neural Network (RSNN with LIF loops + L1 Sparsity).
     - Analytical Role: Full biological synergy integrating LIF spiking, recurrent loops, and L1 population sparsity.

UNIFIED RETURN CONTRACT:
  - Agents A, B, D  ->  (h_rep, logits)          where h_rep: [batch, 32], logits: [batch, 4]
  - Agent C         ->  (h_rep, logits, h_next)   where h_rep = h_next: [batch, 32]
  This allows train.py to execute a clean, unified actor_forward() helper function.
========================================================================================================
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


# ========================================================================================================
# AGENT A: DENSE CONTINUOUS MLP BASELINE
# ========================================================================================================
class AgentA_MLP(nn.Module):
    """
    Agent A: Dense Feedforward Multi-Layer Perceptron (MLP) Baseline.
    
    Architecture:
      - Input Layer: 5-element continuous egocentric distance vector.
      - Hidden Layer: 32 dense continuous units with ReLU activation.
      - Output Layer: 4 continuous logits representing action distribution.
    """
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_rep = torch.relu(self.fc1(x))   # Hidden population state [batch, 32]
        logits = self.fc2(h_rep)           # Policy logits [batch, 4]
        return h_rep, logits


# ========================================================================================================
# AGENT B: FEEDFORWARD SPIKING NEURAL NETWORK (FF-SNN)
# ========================================================================================================
class AgentB_FFSNN(nn.Module):
    """
    Agent B: Feedforward Spiking Neural Network (FF-SNN).
    
    Architecture:
      - Hidden Layer: 32 Leaky Integrate-and-Fire (LIF) spiking units (snnTorch).
      - Temporal Unrolling: T = 20 temporal steps per environment step.
      - Surrogate Gradient: Fast-sigmoid surrogate backpropagation (slope = 25).
      - Membrane Decay: Beta = 0.9.
    """
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4, beta=0.9, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        # Fast-sigmoid surrogate gradient for non-differentiable threshold backprop
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x):
        # Initialize membrane potential tensors at zero
        mem1 = torch.zeros(x.size(0), self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)

        spk1_rec, spk2_rec = [], []
        # Unroll T=20 temporal steps per environment step
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        # Mean firing rate over T=20 steps forms the policy representation
        h_rep  = torch.stack(spk1_rec).mean(dim=0)   # [batch, 32]
        logits = torch.stack(spk2_rec).mean(dim=0)   # [batch, 4]
        return h_rep, logits


# ========================================================================================================
# AGENT C: CONTINUOUS RECURRENT NEURAL NETWORK (RNN)
# ========================================================================================================
class AgentC_RNN(nn.Module):
    """
    Agent C: Continuous Recurrent Neural Network (RNN) Memory Baseline.
    
    Architecture:
      - Hidden Layer: PyTorch nn.RNNCell with 32 units.
      - Recurrence: Persistent hidden state vector h_{t-1} -> h_t passed across environment steps.
    """
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


# ========================================================================================================
# AGENT D: RECURRENT SPIKING NEURAL NETWORK (RSNN)
# ========================================================================================================
class AgentD_RSNN(nn.Module):
    """
    Agent D: Recurrent Spiking Neural Network (RSNN) with L1 Activity Sparsity.
    
    Architecture:
      - Hidden Layer: 32 Recurrent LIF spiking units (snnTorch).
      - Recurrent Feedback: Spikes from step t-1 feedback into hidden layer at step t.
      - Learnable Beta: Membrane decay parameters are learnable.
      - Temporal Unrolling: T = 20 temporal steps per environment step.
    """
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4, beta=0.9, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc_in  = nn.Linear(input_dim, hidden_dim)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

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
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Unified return contract across all agents:
#   A, B, D  ->  (h_rep, logits)          h_rep: [batch, 32]
#   C        ->  (h_rep, logits, h_next)   h_rep = h_next: [batch, 32]
# This keeps train.py clean via a single actor_forward() helper.


# ==========================================
# AGENT A: Dense MLP Baseline
# ==========================================
class AgentA_MLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_rep = torch.relu(self.fc1(x))   # [batch, 32]
        logits = self.fc2(h_rep)           # [batch, 4]
        return h_rep, logits


# ==========================================
# AGENT B: Feedforward SNN
# ==========================================
class AgentB_FFSNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4, beta=0.9, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x):
        mem1 = torch.zeros(x.size(0), self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)

        spk1_rec, spk2_rec = [], []
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        # Mean firing rate over T steps -> policy-relevant population summary
        h_rep  = torch.stack(spk1_rec).mean(dim=0)   # [batch, 32]
        logits = torch.stack(spk2_rec).mean(dim=0)   # [batch, 4]
        return h_rep, logits


# ==========================================
# AGENT C: Recurrent RNN Baseline
# ==========================================
class AgentC_RNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.RNNCell(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        h_rep = self.rnn_cell(x, h_prev)   # [batch, 32]
        logits = self.fc_out(h_rep)         # [batch, 4]
        # h_rep IS h_next for RNNCell; returning it explicitly for train.py clarity
        return h_rep, logits, h_rep


# ==========================================
# AGENT D: Recurrent SNN (RSNN)
# ==========================================
class AgentD_RSNN(nn.Module):
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
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# ==========================================
# AGENT A: Dense MLP Baseline (No Memory)
# ==========================================
class AgentA_MLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


# ==========================================
# AGENT B: Feedforward SNN (Temporal/No Loops)
# ==========================================
class AgentB_FFSNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Changed init_hidden to False for explicit state tracking
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x, num_steps=20):
        # Initialize membrane potentials explicitly to match batch size dynamically
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Ensure states match the device and batch size of input x
        if x.dim() > 1:
            mem1 = torch.zeros(x.size(0), self.fc1.out_features, device=x.device)
            mem2 = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)
            
        spk2_rec = []
        mem2_rec = []
        
        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1) # Explicitly pass and update membrane
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# ==========================================
# AGENT C: Recurrent RNN (Continuous Memory)
# ==========================================
class AgentC_RNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            
        h_next = self.rnn_cell(x, h_prev)
        out = self.fc_out(h_next)
        return out, h_next


# ==========================================
# AGENT D: Recurrent SNN / RSNN (The Mouse Brain Blueprint)
# ==========================================
class AgentD_RSNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=4, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.lif_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False, learn_beta=True)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=False)

    def forward(self, x, num_steps=20):
        mem_rec = torch.zeros(x.size(0), self.fc_rec.out_features, device=x.device)
        mem_out = torch.zeros(x.size(0), self.fc_out.out_features, device=x.device)
        spk_rec_prev = torch.zeros(x.size(0), self.fc_rec.in_features, device=x.device)
        
        hidden_spikes = []
        out_spikes = []
        
        for step in range(num_steps):
            cur_rec = self.fc_in(x) + self.fc_rec(spk_rec_prev)
            spk_rec, mem_rec = self.lif_rec(cur_rec, mem_rec)
            
            cur_out = self.fc_out(spk_rec)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            
            hidden_spikes.append(spk_rec)
            out_spikes.append(spk_out)
            
            spk_rec_prev = spk_rec
            
        return torch.stack(hidden_spikes, dim=0), torch.stack(out_spikes, dim=0)
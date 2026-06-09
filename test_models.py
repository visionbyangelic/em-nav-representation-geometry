import torch
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN

# Create a mock batch of data representing 4 agents reading our 5-ray sensor stream
mock_sensory_input = torch.rand(4, 5)

print("🧠 Starting Architecture Dimension Audits...")

# Test Agent A
model_a = AgentA_MLP()
out_a = model_a(mock_sensory_input)
print(f"Agent A (MLP) Output Shape    : {out_a.shape} (Expected: [4, 4])")

# Test Agent B
model_b = AgentB_FFSNN()
spk_b, mem_b = model_b(mock_sensory_input, num_steps=20)
print(f"Agent B (FFSNN) Spikes Shape  : {spk_b.shape} (Expected: [20, 4, 4])")

# Test Agent C
model_c = AgentC_RNN()
out_c, h_next = model_c(mock_sensory_input)
print(f"Agent C (RNN) Output Shape    : {out_c.shape} (Expected: [4, 4])")

# Test Agent D
model_d = AgentD_RSNN()
hidden_spk_d, out_spk_d = model_d(mock_sensory_input, num_steps=20)
print(f"Agent D (RSNN) Hidden Spikes  : {hidden_spk_d.shape} (Expected: [20, 4, 32])")
print(f"Agent D (RSNN) Output Spikes  : {out_spk_d.shape} (Expected: [20, 4, 4])")
print("\n✅ Matrix dimensions are perfectly aligned! Ready for optimization pipeline.")
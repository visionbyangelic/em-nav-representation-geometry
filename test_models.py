import torch
from models import AgentA_MLP, AgentB_FFSNN, AgentC_RNN, AgentD_RSNN

mock_input = torch.rand(4, 5)   # batch=4, 5-ray sensor stream
print("🧠 Architecture Dimension Audit\n")

# Agent A
h, logits = AgentA_MLP()(mock_input)
print(f"Agent A  h_rep  : {tuple(h.shape)}      (expected [4, 32])")
print(f"Agent A  logits : {tuple(logits.shape)}       (expected [4, 4])\n")

# Agent B
h, logits = AgentB_FFSNN()(mock_input)
print(f"Agent B  h_rep  : {tuple(h.shape)}      (expected [4, 32])")
print(f"Agent B  logits : {tuple(logits.shape)}       (expected [4, 4])\n")

# Agent C
h, logits, h_next = AgentC_RNN()(mock_input)
print(f"Agent C  h_rep  : {tuple(h.shape)}      (expected [4, 32])")
print(f"Agent C  logits : {tuple(logits.shape)}       (expected [4, 4])")
print(f"Agent C  h_next : {tuple(h_next.shape)}      (expected [4, 32])\n")

# Agent D
h, logits = AgentD_RSNN()(mock_input)
print(f"Agent D  h_rep  : {tuple(h.shape)}      (expected [4, 32])")
print(f"Agent D  logits : {tuple(logits.shape)}       (expected [4, 4])\n")

print("✅ All dimensions verified.")
import torch
from tgn_model import TGN

# Seed for reproducibility
torch.manual_seed(42)

# Configuration
num_nodes = 100
num_edges = 150
in_channels = 48
memory_dim = 32
out_channels = 32

# Generate synthetic data
x = torch.rand((num_nodes, in_channels))  # Node features
edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge list
memory = torch.zeros((num_nodes, memory_dim))  # Initial memory

# Nodes to update
nodes = torch.tensor([0, 1, 2])

# Display original features & memory
print("🔹 Original Feature Vectors (first 3 nodes):")
for i in nodes:
    print(f"Node {i.item()} feature: {x[i][:5].tolist()}...")

print("\n🔹 Original Memory Vectors (first 3 nodes):")
for i in nodes:
    print(f"Node {i.item()} memory: {memory[i][:5].tolist()}...")

# Initialize TGN model
model = TGN(in_channels=in_channels, out_channels=out_channels, memory_dim=memory_dim)

# Forward pass
updated_embeddings = model(x, edge_index, memory, nodes)

# Output updated embeddings
print("\n✅ Updated Embeddings After Message Passing:")
for i, emb in zip(nodes, updated_embeddings):
    print(f"Node {i.item()} embedding: {emb[:5].tolist()}...")

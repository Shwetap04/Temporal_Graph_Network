import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class TGNMessagePassing(MessagePassing):
    """
    A simplified message-passing layer inspired by the Temporal Graph Network (TGN) model.
    Each node combines its feature and memory to send messages to neighbors.
    """
    def __init__(self, in_channels, out_channels, memory_dim):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + memory_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, memory, nodes):
        # Subset node features and memory for selected nodes
        x_sub = x[nodes]
        memory_sub = memory[nodes]

        # Filter edge_index to only edges between nodes in `nodes`
        edge_index = self._filter_edge_index(edge_index, nodes)

        # Remap node indices to local ones
        node_map = {int(node): i for i, node in enumerate(nodes.tolist())}
        edge_index = torch.stack([
            torch.tensor([node_map[int(i)] for i in edge_index[0]]),
            torch.tensor([node_map[int(i)] for i in edge_index[1]])
        ])

        return self.propagate(edge_index, x=x_sub, memory=memory_sub)

    def message(self, x_j, memory_j):
        return torch.cat([x_j, memory_j], dim=-1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)

    def _filter_edge_index(self, edge_index, nodes):
        node_set = set(nodes.tolist())
        src, dst = edge_index
        mask = [(int(s) in node_set and int(d) in node_set) for s, d in zip(src, dst)]
        return edge_index[:, torch.tensor(mask, dtype=torch.bool)]

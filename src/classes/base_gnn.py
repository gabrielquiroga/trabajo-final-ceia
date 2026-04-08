import torch.nn as nn
from torch_geometric.nn import SAGEConv

# =====================================================================
# 2. GNN Base (Homogénea)
# =====================================================================
class BaseGNN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # SAGEConv procesa grafos agregando la media de los vecinos.
        # (-1, -1) indica "lazy initialization" (infiere dinámicamente el tamaño de entrada)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        return x
import torch.nn as nn
from torch_geometric.nn import SAGEConv

# =====================================================================
# 2. GNN Base (Homogénea)
# =====================================================================
# Dos capas SAGEConv apiladas con una activación GELU en el medio. GraphSAGE agrega los features de los
# vecinos de cada nodo mediante la fórmula h_v' = W · CONCAT(h_v, MEAN_{u∈N(v)} h_u). El (-1, -1) de lazy
# init es fundamental: como esta GNN se va a convertir a heterogénea con to_hetero, los tamaños reales de
# entrada se conocerán solo al momento del primer forward, cuando se sabe qué tipos de nodos existen y
# cuántos features tienen.

class BaseGNN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # SAGEConv procesa grafos agregando la media de los vecinos. Para cada nodo v, combina su propio
        # embedding con el promedio de sus vecinos y los proyecta linealmente
        # (-1, -1) indica "lazy initialization" (infiere dinámicamente el tamaño de entrada)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        # GELU(x) = x · Φ(x), donde Φ es la CDF gaussiana. Es una versión suave de ReLU que permite
        # gradientes pequeños para x negativos, lo que mejora el entrenamiento en arquitecturas profundas.
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        return x
import torch
import torch.nn as nn
from torch_geometric.nn import to_hetero

from sinusoidal_position_embeddings import SinusoidalPositionEmbeddings
from base_gnn import BaseGNN


# =====================================================================
# 3. Arquitectura Principal: GNN + Diffusion
# =====================================================================
class InstantPolicyModel(nn.Module):
    def __init__(self, metadata, node_features=2, hidden_dim=64):
        super().__init__()
        
        # A. Expansión inicial de features [x, y] -> [hidden_dim]
        self.action_emb = nn.Linear(node_features, hidden_dim)
        self.context_emb = nn.Linear(node_features, hidden_dim)

        # B. MLP para procesar el Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # C. Conversión de la GNN base a Heterogénea usando tu metadata
        base_gnn = BaseGNN(hidden_dim)
        self.gnn = to_hetero(base_gnn, metadata=metadata, aggr='sum')

        # D. MLP Predictor de Ruido (Cabezal final)
        # La entrada será la concatenación de la salida de la GNN y el Tiempo (hidden_dim * 2)
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_features) # Predice un vector [ruido_x, ruido_y]
        )

    def forward(self, hetero_data, timestep):
        # NOTA CLAVE: En la etapa de entrenamiento, hetero_data['action'].x 
        # no contendrá la trayectoria limpia, sino la trayectoria CON RUIDO aplicado.
        
        # 1. Proyectar las coordenadas [x, y] al espacio latente
        x_dict = {
            'context': self.context_emb(hetero_data['context'].x),
            'action': self.action_emb(hetero_data['action'].x)
        }

        # 2. Paso de Mensajes (Message Passing) en el grafo heterogéneo
        # Utiliza los tensores de aristas definidos en tu notebook anterior
        node_embeddings = self.gnn(x_dict, hetero_data.edge_index_dict)

        # Extraemos solo los features actualizados de los nodos que queremos predecir
        action_features = node_embeddings['action']

        # 3. Procesar el tiempo
        # timestep debe ser un tensor de forma [batch_size]
        t_emb = self.time_mlp(timestep)

        # ALINEACIÓN DE BATCHES (PyTorch Geometric agrupa grafos apilando nodos)
        # Si usamos un DataLoader, necesitamos expandir el vector de tiempo escalar 
        # para que coincida con cada nodo individual perteneciente a ese grafo en el batch.
        if hasattr(hetero_data['action'], 'batch') and hetero_data['action'].batch is not None:
            batch_idx = hetero_data['action'].batch
            t_emb = t_emb[batch_idx] 
        else:
            # Si estamos inferiendo un solo grafo sin DataLoader
            t_emb = t_emb.expand(action_features.shape[0], -1)

        # 4. Fusionar contexto del grafo con el tiempo y predecir
        fused_features = torch.cat([action_features, t_emb], dim=-1)
        predicted_noise = self.noise_predictor(fused_features)

        return predicted_noise
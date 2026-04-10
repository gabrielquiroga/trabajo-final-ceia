import torch
import torch.nn as nn
from torch_geometric.nn import to_hetero

from src.classes.sinusoidal_position_embeddings import SinusoidalPositionEmbeddings
from src.classes.base_gnn import BaseGNN


# =====================================================================
# 3. Arquitectura Principal: GNN + Diffusion
# =====================================================================
# El __init__ construye 4 submódulos:
# A) Dos nn.Linear(2, hidden_dim) separados para proyectar coordenadas [x, y] al espacio latente. Son independientes
# para que el modelo aprenda representaciones distintas según si el nodo es de contexto (escena) o de acción (trayectoria).
# B) Un nn.Sequential que combina el embedding sinusoidal con dos capas lineales (con cuello de botella 64→128→64),
# dando al modelo capacidad de transformar no-linealmente la representación del tiempo.
# C) to_hetero(base_gnn, metadata, aggr='sum') replica internamente los pesos de BaseGNN para cada tipo de arista del
# grafo, creando una GNN que puede procesar relaciones heterogéneas. Con aggr='sum', cuando un nodo recibe mensajes de
# múltiples tipos de aristas, los suma.
# D) El noise_predictor es un MLP que toma la concatenación del embedding de grafo y del tiempo [128-dim] y lo proyecta de
# vuelta a [2] (el ruido predicho en x e y).
#
# En el forward, el momento más delicado es la alineación de batches: PyTorch Geometric representa un batch de grafos
# apilando todos sus nodos en un solo tensor, y el vector .batch actúa como un índice que mapea cada nodo al grafo original.
# Con t_emb[batch_idx] se "expande" el embedding de tiempo del grafo correcto a cada uno de sus nodos, igualando las
# dimensiones para poder concatenar.

class InstantPolicyModel(nn.Module):
    def __init__(self, metadata, node_features=2, hidden_dim=64):
        super().__init__()
        
        # A. Expansión inicial de features [x, y] -> [hidden_dim]
        # Dos capas nn.Linear(2, 64) independientes para nodos de tipo context y action. Aunque la estructura es la misma,
        # tienen pesos distintos: el modelo puede aprender representaciones latentes diferentes según el rol del nodo.
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
        # NOTA CLAVE: En la etapa de entrenamiento, hetero_data['action'].x no contendrá la
        # trayectoria limpia, sino la trayectoria CON RUIDO aplicado.
        # Durante el entrenamiento de difusión, el input action.x es la trayectoria ruidosa x_t = x_0 + ε·σ_t.
        # La red aprende a predecir el ruido ε que fue añadido, NO la trayectoria limpia directamente. Esto
        # es el objetivo estándar del noise prediction (ε-parametrization) de DDPM.
        
        # 1. Proyectar las coordenadas [x, y] al espacio latente
        x_dict = {
            'context': self.context_emb(hetero_data['context'].x),
            'action': self.action_emb(hetero_data['action'].x)
        }

        # 2. Paso de Mensajes (Message Passing) en el grafo heterogéneo
        # Utiliza los tensores de aristas definidos en el preprocesamiento.
        # La GNN heterogénea itera sobre cada tipo de arista en edge_index_dict. Para cada
        # arista (tipo_origen, tipo_relación, tipo_destino), corre una instancia independiente
        # de SAGEConv. Los mensajes que llegan de distintos tipos se combinan con aggr='sum'.
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
            # El vector .batch indica a qué grafo pertenece cada nodo (ej. [0,0,0,1,1,1,2,2]).
            # Se usa como índice para replicar el embedding de tiempo correcto a cada nodo.
            t_emb = t_emb[batch_idx] 
        else:
            # Si estamos inferiendo un solo grafo sin DataLoader
            t_emb = t_emb.expand(action_features.shape[0], -1)

        # 4. Fusionar contexto del grafo con el tiempo y predecir
        fused_features = torch.cat([action_features, t_emb], dim=-1)
        predicted_noise = self.noise_predictor(fused_features)

        return predicted_noise
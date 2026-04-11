import torch
import torch.nn as nn
import math

# =====================================================================
# 1. Módulo de Embedding de Tiempo (Sinusoidal)
# =====================================================================
# Convierte un timestep escalar t (un entero que indica en qué paso del proceso de difusión estamos) en un vector denso de dimensión dim. La
# fórmula usa exp(-i · log(10000) / (half_dim - 1)) que es algebraicamente idéntica a 1 / 10000^(i/(d/2-1)), generando half_dim frecuencias
# logarítmicamente espaciadas. Luego computa seno y coseno de t modulado por cada frecuencia, y los concatena. El resultado es que timesteps
# cercanos producen embeddings similares, pero cada timestep tiene una firma única.
# El truco de broadcasting time[:, None] * embeddings[None, :] hace un producto externo entre el batch de timesteps [B] y el vector de
# frecuencias [half_dim], produciendo [B, half_dim] sin ningún loop.

# nn.Module es la clase base para todos los módulos de PyTorch, lo que nos permite definir capas personalizadas.
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # Mismo constructor, pero agregando el tamaño del vector de embedding

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2 # Se separa la dimensión en dos partes para aplicar sin y cos
        embeddings = math.log(10000) / (half_dim - 1) # log(10000) / (dim/2 - 1) es la fórmula estándar para calcular las frecuencias de los embeddings sinusoidales
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) #e^(-log(10000) / (dim/2 - 1) * i) para i en [0, dim/2-1]
        # Broadcasting para multiplicar el tiempo escalar por las frecuencias
        # time[:, None] convierte el vector de tiempo de forma [batch_size] a [batch_size, 1], y embeddings[None, :] convierte el vector de frecuencias
        # de forma [half_dim] a [1, half_dim], lo que permite la multiplicación elemento a elemento resultando en un tensor de forma [batch_size, half_dim]
        embeddings = time.float()[:, None] * embeddings[None, :] 
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
        # Escalar -> Vector[dim/2] -> Tensor[Batch, dim/2] -> Tensor[Batch, dim]
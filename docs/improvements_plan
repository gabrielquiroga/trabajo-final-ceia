# Revisión Integral del Proyecto "Instant Policy" — v2 (Actualizado)

> [!NOTE]
> Esta versión corrige observaciones del análisis previo basándose en los comentarios del autor y en una re-revisión del código actualizado.

## 1. Resumen del Estado Actual

| Componente | Estado | Archivos |
|---|---|---|
| Generación de trayectorias | ✅ Funcional | [trajectory_generator.py](file:///c:/CEIA/trabajo-final-ceia/src/classes/trajectory_generator.py), [examples_generation_v2.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/examples_generation_v2.ipynb) |
| Generación de dataset (HDF5) | ✅ Funcional (v2) | dataset_instant_policy_2d_v2.h5 |
| Normalización conjunta | ✅ Funcional | `normalize_pair()` en el notebook |
| Grafos Heterogéneos (3 tipos de arista) | ✅ Funcional | `build_hetero_graph()` en el notebook |
| Embeddings sinusoidales | ✅ Funcional | [sinusoidal_position_embeddings.py](file:///c:/CEIA/trabajo-final-ceia/src/classes/sinusoidal_position_embeddings.py) |
| GNN Base (SAGEConv) | ✅ Funcional | [base_gnn.py](file:///c:/CEIA/trabajo-final-ceia/src/classes/base_gnn.py) |
| DDPM Scheduler | ✅ Funcional | [DDPMScheduler.py](file:///c:/CEIA/trabajo-final-ceia/src/classes/DDPMScheduler.py) |
| Modelo principal | ✅ Funcional | [instant_policy_model.py](file:///c:/CEIA/trabajo-final-ceia/src/classes/instant_policy_model.py) |
| Loop de entrenamiento | ✅ Funcional | [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb) |
| Loop de inferencia (sampling) | ✅ Funcional (con observación menor) | [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb) |
| Separación train/test | ❌ Faltante | — |
| Métricas de evaluación | ❌ Faltante | — |
| Visualización comparativa | ⚠️ Parcial | — |
| Guardado/carga de modelo | ❌ Faltante | — |

---

## 2. Problemas Encontrados y Recomendaciones

### 2.1 🟢 Varianza en el paso reverso DDPM — Aclaración (Corregido)

> [!NOTE]
> Tras re-analizar, la fórmula actual **es correcta**. La confusión era autoimpuesta; a continuación se documenta el razonamiento para que quede claro.

En el código de inferencia:

```python
sigma_t = torch.sqrt(1.0 - alpha_t)
```

Dado que en DDPM se define `alpha_t = 1 - beta_t`, entonces `1 - alpha_t = beta_t`. Por lo tanto:

```
sigma_t = sqrt(1 - alpha_t) = sqrt(beta_t)
```

Esto corresponde exactamente a la **varianza fija** (`σ²_t = β_t`) del paper original de Ho et al. (2020). Es una de las dos opciones válidas de varianza para el paso reverso:

1. **Varianza fija (la que estás usando):** `σ²_t = β_t` — más simple, perfectamente válida
2. **Varianza posterior:** `σ²_t = β̃_t = β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)` — teóricamente óptima, da muestras ligeramente mejores

**Recomendación**: El código actual es correcto. Para documentarlo mejor, se sugiere agregar un comentario explícito:

```python
# Varianza fija del DDPM: σ²_t = β_t (Ho et al., 2020, Eq. 11)
# Nota: sqrt(1 - alpha_t) = sqrt(beta_t) por definición de alpha_t = 1 - beta_t
sigma_t = torch.sqrt(1.0 - alpha_t)
```

Opcionalmente, se puede mejorar la calidad de las muestras generadas implementando la varianza posterior:

```python
# Varianza posterior (teóricamente óptima):
alpha_cumprod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
beta_t = scheduler.betas[t].to(device)
sigma_t = torch.sqrt(beta_t * (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t))
```

---

### 2.2 ✅ Normalización de datos — Ya implementada (Confirmado)

La función `normalize_pair()` ya está implementada y es llamada dentro de `build_hetero_graph()` (línea 100 del notebook). La normalización es **conjunta** y **uniforme** (usa el máximo rango de ambos ejes), lo cual es correcto para la tarea.

No se necesita normalización adicional en el loop de carga del dataset. El comentario viejo en el notebook (`# NORMALIZAR LAS COORDENADAS SI NO LO ESTÁN!`) es un vestigio de una versión anterior y debería eliminarse para evitar confusión.

---

### 2.3 ✅ Aristas context ↔ context — Ya implementadas (Confirmado)

El código en `build_hetero_graph()` (líneas 156-167) implementa correctamente aristas secuenciales bidireccionales para los nodos `context`. Esto fue verificado en la salida del notebook:

```
(context, to, context)={ edge_index=[2, 14] }
```

Esto corresponde a 8 nodos context × 2 (bidireccional) - 2 (extremos) = 14 aristas, lo cual es correcto.

---

### 2.4 🟡 Separación de datos train/test — Problema real

> [!IMPORTANT]
> El problema principal aquí **no** es la inconsistencia técnica del batch handling (que funciona correctamente), sino que **se usan los mismos datos para entrenamiento e inferencia**. Sin un conjunto de test separado, no hay forma de medir si el modelo realmente generaliza.

Actualmente:
- Se entrena con **todas** las 2000 demos del dataset
- Se hace inferencia sobre grafos que el modelo ya "vio" durante el entrenamiento
- Esto puede dar resultados engañosamente buenos sin que haya verdadera generalización

**Recomendación** (dos opciones):

1. **Split train/val/test**: Dividir las 2000 demos existentes (ej: 70/15/15)
2. **Generar datos de test nuevos**: Crear un segundo dataset con `examples_generation_v2.ipynb` usando parámetros aleatorios diferentes, garantizando que son trayectorias que el modelo nunca vio

La opción 2 es más robusta conceptualmente para un proyecto final, ya que demuestra generalización a trayectorias genuinamente nuevas.

---

### 2.5 🟡 dtype inconsistente en timesteps

En la inferencia (`ddpm_reverse_step`):
```python
t_tensor = torch.tensor([t], device=device, dtype=torch.long)
```

Pero el `SinusoidalPositionEmbeddings` realiza:
```python
embeddings = time[:, None] * embeddings[None, :]  # multiplicación float
```

El `time[:, None]` es `torch.long`, y se multiplica con un tensor float. PyTorch hace casting implícito, pero es mejor ser explícito:

```python
# Opción 1: cambiar el dtype al crear el tensor
t_tensor = torch.tensor([t], device=device, dtype=torch.float32)

# Opción 2: castear dentro del embedding
embeddings = time.float()[:, None] * embeddings[None, :]
```

**Impacto**: Bajo (funciona gracias al casting automático de PyTorch), pero es buena práctica corregirlo.

---

### 2.6 🟡 `num_action_nodes` inconsistente entre entrenamiento e inferencia

En `generate_trajectory()`:
```python
def generate_trajectory(model, scheduler, context_graph, num_action_nodes=50, ...):
```

Pero en el dataset v2, los grafos tienen tamaños de action nodes variables (se observa `action={ x=[100, 2] }` en la salida). Si el modelo fue entrenado con 100 nodos action, la inferencia con 50 producirá resultados inconsistentes.

**Recomendación**: El `num_action_nodes` debería coincidir con el número de nodos action usado en el entrenamiento, o al menos pasarse como parámetro explícito desde los datos de test.

---

### 2.7 🟢 GNN de 2 capas — Observación menor

La `BaseGNN` con 2 capas SAGEConv puede ser suficiente para la PoC con trayectorias 2D simples. Si los resultados de generación no son satisfactorios, considerar:
- Agregar 1-2 capas más
- Agregar residual connections (`x = self.conv1(x, edge_index) + x_original`)
- Aumentar `hidden_dim` de 64 a 128

---

## 3. Componentes Faltantes para Finalizar el Proyecto

### 3.1 ❌ Generación de datos de test separados
- Crear un segundo dataset con nuevas trayectorias (diferentes semillas/parámetros)
- O implementar un split train/val/test del dataset existente
- Indispensable para medir generalización

### 3.2 ❌ Métricas de evaluación cuantitativa
Para un proyecto final completo se necesitan al menos 2-3 de estas métricas:
- **Mean Squared Error (MSE)**: entre trayectoria generada y target ground truth (post-normalización)
- **Chamfer Distance**: distancia promedio al punto más cercano entre las dos trayectorias
- **Suavidad (smoothness)**: magnitud del "jerk" (derivada segunda discreta), para verificar que la trayectoria no tiene saltos bruscos
- **Fréchet Distance**: similitud geométrica de trayectorias considerando el orden

### 3.3 ❌ Guardado y carga del modelo
```python
# Guardar
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'metadata': metadata,
}, 'checkpoint.pth')

# Cargar
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

Esto permite:
- Retomar entrenamiento si se interrumpe
- Evaluar sin re-entrenar
- Demostrar el modelo en la presentación final

### 3.4 ❌ Validación durante entrenamiento
- Calcular validation loss cada N epochs
- Implementar early stopping si la val loss no mejora
- Separar al menos un 15-20% de datos para validación

### 3.5 ⚠️ Visualización comparativa robusta
Ya hay visualizaciones parciales (proceso de denoising). Para el proyecto final conviene una función que muestre en el mismo plot:
- Contexto (demo experta) en un color
- Target (ground truth) en otro
- Trayectoria generada por el modelo en otro
- Con leyenda y título indicando el tipo de trayectoria

### 3.6 ⚠️ Limpieza del notebook
- Eliminar comentarios obsoletos (ej: `# NORMALIZAR LAS COORDENADAS SI NO LO ESTÁN!`)
- Mover funciones reutilizables a módulos Python (`src/training/`, `src/evaluation/`)
- Agregar docstrings a las funciones del notebook

### 3.7 ⚠️ Hiperparámetros por afinar
| Parámetro | Valor Actual | Recomendación |
|---|---|---|
| `num_timesteps` | 100 | OK para PoC, probar 200 si hay tiempo |
| `hidden_dim` | 64 | Probar 128 si los resultados son pobres |
| `k_neighbors` | 5 | Probar 3 y 7 para ver impacto |
| `learning_rate` | 1e-3 | Agregar LR scheduler (CosineAnnealing) |
| `epochs` | 50 | Suficiente si la loss se estabiliza |
| `batch_size` | 8 | OK para CPU; 32+ para GPU |
| Capas GNN | 2 | Probar 3 si la calidad no es buena |

---

## 4. Plan de Acción Priorizado

### Fase 1: Correcciones rápidas (1-2 horas)
1. ~~Corregir dtype de timesteps a `float32` en inferencia~~
2. ~~Corregir `num_action_nodes` para que coincida con los datos de entrenamiento~~
3. ~~Eliminar comentarios obsoletos del notebook~~
4. ~~Documentar la fórmula de varianza~~

### Fase 2: Datos y evaluación (3-4 horas)
5. Implementar split train/val o generar dataset de test separado
6. Implementar 2-3 métricas de evaluación cuantitativa
7. Crear función de visualización comparativa (context + target + generada)

### Fase 3: Robustez del entrenamiento (2-3 horas)
8. Implementar guardado/carga de checkpoints
9. Agregar validation loss al loop de entrenamiento
10. Opcional: agregar LR scheduler

### Fase 4: Polish final (1-2 horas)
11. Refactorizar funciones a módulos Python
12. Ajuste fino de hiperparámetros
13. Documentación final del proyecto

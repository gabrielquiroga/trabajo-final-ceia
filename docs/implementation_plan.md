# Modificaciones Requeridas — Proyecto Instant Policy

A continuación están **todas** las modificaciones necesarias, organizadas por problema → solución, con el código exacto.

Aplicalas en el orden que quieras; son independientes entre sí (salvo que se indique lo contrario).

---

## 1. Corregir dtype de timesteps en inferencia

**Problema**: En `ddpm_reverse_step`, el tensor de timestep se crea con `dtype=torch.long` (entero), pero el `SinusoidalPositionEmbeddings` opera en punto flotante (`sin`, `cos`, multiplicaciones con floats). Aunque PyTorch hace casting implícito, es mejor ser explícito para evitar warnings o errores sutiles en futuras versiones.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb)

**Código actual** (dentro de `ddpm_reverse_step`):
```python
t_tensor = torch.tensor([t], device=device, dtype=torch.long)
```

**Código corregido**:
```python
t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
```

> [!NOTE]
> Este cambio también debe aplicarse en el training loop si allí los timesteps se pasan al modelo con dtype `long`. Verificar que `timesteps` en `train_diffusion_policy` se cree con `.float()` antes de pasar al modelo, **o** agregar un `.float()` dentro del `SinusoidalPositionEmbeddings.forward()` (ver modificación 1b).

### 1b. Alternativa: Hacer el cast dentro del embedding (más robusto)

**Archivo**: [sinusoidal_position_embeddings.py](file:///c:/CEIA/trabajo-final-ceia/src/classes/sinusoidal_position_embeddings.py)

**Código actual** (línea ~29):
```python
embeddings = time[:, None] * embeddings[None, :]
```

**Código corregido**:
```python
embeddings = time.float()[:, None] * embeddings[None, :]
```

> [!TIP]
> Esta alternativa es más defensiva: no importa qué dtype llegue, siempre funcionará correctamente. Recomiendo aplicar **ambas** correcciones.

---

## 2. Documentar la fórmula de varianza en el paso reverso

**Problema**: El comentario actual sobre `sigma_t` es ambiguo ("simplificación estándar"). Esto genera confusión sobre cuál de las dos varianzas de DDPM se está utilizando. La fórmula es correcta, pero debe documentarse claramente.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb)

**Código actual** (dentro de `ddpm_reverse_step`, bloque `if t > 0`):
```python
    if t > 0:
        z = torch.randn_like(current_x)
        # Una varianza común para sigma_t es simplemente sqrt(1 - alpha_t)
        # o usar la varianza posterior calculada, aquí usamos una simplificación estándar
        sigma_t = torch.sqrt(1.0 - alpha_t)
        x_t_minus_1 = mean_x_t_minus_1 + sigma_t * z
```

**Código corregido**:
```python
    if t > 0:
        z = torch.randn_like(current_x)
        # Varianza fija del DDPM (Ho et al., 2020, Eq. 11):
        #   σ²_t = β_t
        # Como alpha_t = 1 - beta_t, entonces:
        #   sqrt(1 - alpha_t) = sqrt(beta_t)
        # Esta es una de las dos opciones válidas del paper original.
        # La otra (varianza posterior) sería:
        #   σ²_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        sigma_t = torch.sqrt(1.0 - alpha_t)
        x_t_minus_1 = mean_x_t_minus_1 + sigma_t * z
```

---

## 3. Corregir `num_action_nodes` en `generate_trajectory`

**Problema**: El default `num_action_nodes=50` no coincide con los datos del dataset v2, donde los grafos tienen 100 nodos action (como se observa en la salida del notebook: `action={ x=[100, 2] }`). Esto causa una inconsistencia entre entrenamiento e inferencia.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb)

**Código actual**:
```python
def generate_trajectory(model, scheduler, context_graph, num_action_nodes=50, device='cpu'):
```

**Código corregido** (opción A — cambiar el default):
```python
def generate_trajectory(model, scheduler, context_graph, num_action_nodes=100, device='cpu'):
```

**Código corregido** (opción B — obtener el valor del grafo, más robusto):
```python
def generate_trajectory(model, scheduler, context_graph, num_action_nodes=None, device='cpu'):
    """
    Toma un grafo de contexto y genera una trayectoria desde cero
    mediante el proceso reverso de DDPM.
    """
    model.eval()

    # Si no se especifica, usar el mismo número de nodos action que tiene el grafo
    if num_action_nodes is None:
        num_action_nodes = context_graph['action'].x.shape[0]

    current_x = torch.randn((num_action_nodes, 2), device=device)
    # ... resto de la función sin cambios
```

> [!IMPORTANT]
> La opción B es más robusta porque se adapta automáticamente al número de nodos del grafo de entrada, sin depender de un valor hardcodeado.

---

## 4. Eliminar comentario obsoleto del loop de carga

**Problema**: El código de carga del dataset tiene un comentario que dice `# NORMALIZAR LAS COORDENADAS SI NO LO ESTÁN!`, pero la normalización ya se realiza dentro de `build_hetero_graph` → `normalize_pair`. El comentario es un vestigio de una versión anterior y genera confusión.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb)

**Código actual** (celda de ejecución del entrenamiento):
```python
dataset_list = []
for key, graph in iter_dataset_as_graphs(HDF5_PATH):
    # NORMALIZAR LAS COORDENADAS SI NO LO ESTÁN!
    # Ej: graph['action'].x = (graph['action'].x - mean) / std
    dataset_list.append(graph)
```

**Código corregido**:
```python
dataset_list = []
for key, graph in iter_dataset_as_graphs(HDF5_PATH):
    # La normalización ya se aplica dentro de build_hetero_graph -> normalize_pair()
    dataset_list.append(graph)
```

---

## 5. Implementar separación train/test

**Problema**: Actualmente se usan las mismas 2000 demos para entrenamiento e inferencia. Sin datos de test separados, no hay forma de medir si el modelo realmente generaliza a trayectorias que nunca vio.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb), celda de carga del dataset.

**Código a agregar** (reemplazar la celda de carga actual):
```python
import random

# 1. Cargar todos los grafos (la normalización ya se aplica en build_hetero_graph)
all_graphs = []
for key, graph in iter_dataset_as_graphs(HDF5_PATH):
    all_graphs.append(graph)

# 2. Separar en train/val/test (70/15/15)
random.seed(42)  # Para reproducibilidad
random.shuffle(all_graphs)

n_total = len(all_graphs)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

train_list = all_graphs[:n_train]
val_list = all_graphs[n_train:n_train + n_val]
test_list = all_graphs[n_train + n_val:]

print(f"✅ Dataset dividido: {len(train_list)} train / {len(val_list)} val / {len(test_list)} test")
```

> [!IMPORTANT]
> Luego, en la llamada a `train_diffusion_policy`, pasar `train_list` en lugar de `dataset_list`. Ver la modificación #6 para la integración con validation.

---

## 6. Agregar validation loss al loop de entrenamiento

**Problema**: Sin validation loss, no hay forma de detectar overfitting ni de saber cuándo parar el entrenamiento.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb)

**Función `train_diffusion_policy` actualizada**:
```python
def train_diffusion_policy(model, train_list, val_list=None, epochs=100, batch_size=32, lr=1e-3, device='cpu'):
    """
    Entrena el modelo de difusión con soporte para validation loss.
    """
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False) if val_list else None

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = DDPMScheduler(num_timesteps=100)

    model = model.to(device)

    train_loss_history = []
    val_loss_history = []

    print(f"Iniciando entrenamiento en {device}...")
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            clean_actions = batch['action'].x
            noise = torch.randn_like(clean_actions)

            num_graphs = batch.num_graphs
            timesteps = torch.randint(0, scheduler.num_timesteps, (num_graphs,), device=device).long()
            node_timesteps = timesteps[batch['action'].batch]

            noisy_actions = scheduler.add_noise(clean_actions, noise, node_timesteps)
            batch['action'].x = noisy_actions

            predicted_noise = model(batch, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- VALIDATION ---
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    clean_actions = batch['action'].x
                    noise = torch.randn_like(clean_actions)

                    num_graphs = batch.num_graphs
                    timesteps = torch.randint(0, scheduler.num_timesteps, (num_graphs,), device=device).long()
                    node_timesteps = timesteps[batch['action'].batch]

                    noisy_actions = scheduler.add_noise(clean_actions, noise, node_timesteps)
                    batch['action'].x = noisy_actions

                    predicted_noise = model(batch, timesteps)
                    loss = F.mse_loss(predicted_noise, noise)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f}")

    return train_loss_history, val_loss_history
```

**Actualizar la llamada**:
```python
train_history, val_history = train_diffusion_policy(
    model=model,
    train_list=train_list,
    val_list=val_list,
    epochs=50,
    batch_size=8,
    lr=1e-3,
    device=device
)

# Graficar ambas curvas
import matplotlib.pyplot as plt
plt.plot(train_history, label='Train Loss')
if val_history:
    plt.plot(val_history, label='Val Loss')
plt.title("Curva de Aprendizaje - Diffusion Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()
```

---

## 7. Agregar guardado/carga de modelo (checkpointing)

**Problema**: Sin checkpointing, si se interrumpe el entrenamiento se pierde todo el progreso. Además, es necesario para demostrar el modelo sin re-entrenar.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb) — agregar en una celda nueva después del entrenamiento.

**Código para guardar**:
```python
import os

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Guarda el estado del modelo y del optimizador."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metadata': model.gnn.metadata() if hasattr(model, 'gnn') else None,
    }, path)
    print(f"✅ Checkpoint guardado en {path}")

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Carga el estado del modelo desde un checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint.get('val_loss', None)
    print(f"✅ Checkpoint cargado desde {path} (epoch {epoch})")
    return epoch, train_loss, val_loss

# Ejemplo de uso después del entrenamiento:
# save_checkpoint(model, optimizer, epoch, train_history[-1], val_history[-1] if val_history else None,
#                 os.path.join(CHECKPOINT_DIR, "best_model.pth"))
```

> [!TIP]
> Para un uso más avanzado, podés integrar el guardado dentro del loop de entrenamiento (modificación #6) para guardar automáticamente cuando la val loss mejore.

---

## 8. Agregar métricas de evaluación cuantitativa

**Problema**: Sin métricas cuantitativas, no hay forma objetiva de medir la calidad de las trayectorias generadas.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb) — agregar en una celda nueva.

```python
import numpy as np

def mse_trajectories(generated, target):
    """
    Mean Squared Error entre dos trayectorias.
    Ambas deben ser arrays de shape (N, 2).
    """
    return np.mean((generated - target) ** 2)

def chamfer_distance(traj_a, traj_b):
    """
    Chamfer Distance entre dos conjuntos de puntos 2D.
    Para cada punto en A, busca el más cercano en B y viceversa.
    """
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(traj_a, traj_b)
    # Para cada punto en A, la distancia mínima a B
    min_a_to_b = np.mean(np.min(dist_matrix, axis=1))
    # Para cada punto en B, la distancia mínima a A
    min_b_to_a = np.mean(np.min(dist_matrix, axis=0))
    return (min_a_to_b + min_b_to_a) / 2.0

def smoothness(trajectory):
    """
    Mide la suavidad de una trayectoria como la magnitud promedio
    de la segunda derivada discreta (aceleración / jerk).
    Valores más bajos = trayectoria más suave.
    """
    if len(trajectory) < 3:
        return 0.0
    # Primera derivada (velocidad)
    velocity = np.diff(trajectory, axis=0)
    # Segunda derivada (aceleración)
    acceleration = np.diff(velocity, axis=0)
    # Magnitud promedio de la aceleración
    return np.mean(np.linalg.norm(acceleration, axis=1))

def evaluate_model(model, scheduler, test_graphs, device='cpu', num_samples=10):
    """
    Evalúa el modelo sobre un subconjunto de grafos de test.
    Retorna las métricas promedio.
    """
    mse_scores = []
    chamfer_scores = []
    smoothness_scores = []

    for i, graph in enumerate(test_graphs[:num_samples]):
        # Generar trayectoria
        generated, _ = generate_trajectory(model, scheduler, graph, device=device)
        generated_np = generated.numpy()

        # Target (ground truth, ya normalizado dentro del grafo)
        target_np = graph['action'].x.numpy()

        # Calcular métricas
        mse_scores.append(mse_trajectories(generated_np, target_np))
        chamfer_scores.append(chamfer_distance(generated_np, target_np))
        smoothness_scores.append(smoothness(generated_np))

    results = {
        'MSE (promedio)': np.mean(mse_scores),
        'Chamfer Distance (promedio)': np.mean(chamfer_scores),
        'Smoothness (promedio)': np.mean(smoothness_scores),
    }

    print("=" * 50)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 50)
    for metric, value in results.items():
        print(f"  {metric}: {value:.6f}")
    print("=" * 50)

    return results
```

**Uso**:
```python
# Después de entrenar y cargar el modelo:
scheduler = DDPMScheduler(num_timesteps=100)
results = evaluate_model(model, scheduler, test_list, device=device, num_samples=20)
```

> [!NOTE]
> Para que esto funcione, necesitás tener separados los datos de test (ver modificación #5). La función `generate_trajectory` debe estar definida previamente (ver modificación #3).

---

## 9. Agregar visualización comparativa

**Problema**: Falta una función que muestre en el mismo plot el contexto, el target y la trayectoria generada, para inspección visual de la calidad.

**Archivo**: [heterogeneous_graphs.ipynb](file:///c:/CEIA/trabajo-final-ceia/src/heterogeneous_graphs.ipynb) — agregar en una celda nueva.

```python
def plot_comparison(graph, generated_trajectory, title="Comparación de Trayectorias"):
    """
    Visualiza en un mismo plot:
      - Context (demo): puntos rojos
      - Target (ground truth): línea verde
      - Generated (modelo): línea azul punteada
    """
    import matplotlib.pyplot as plt

    ctx = graph['context'].x.numpy()
    target = graph['action'].x.numpy()
    generated = generated_trajectory.numpy() if hasattr(generated_trajectory, 'numpy') else generated_trajectory

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')

    # Context (demo)
    ax.plot(ctx[:, 0], ctx[:, 1], 'o-', color='tomato', markersize=5, linewidth=1.5, label='Context (demo)', alpha=0.8)

    # Target (ground truth)
    ax.plot(target[:, 0], target[:, 1], '-', color='forestgreen', linewidth=2, label='Target (ground truth)', alpha=0.8)

    # Generated
    ax.plot(generated[:, 0], generated[:, 1], '--', color='royalblue', linewidth=2, label='Generated (modelo)', alpha=0.8)

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

**Uso**:
```python
# Evaluar y visualizar algunos ejemplos del test set
scheduler = DDPMScheduler(num_timesteps=100)

for i in range(5):  # Visualizar 5 ejemplos
    graph = test_list[i]
    generated, history = generate_trajectory(model, scheduler, graph, device=device)
    plot_comparison(graph, generated, title=f"Test sample #{i+1}")
```

---

## Resumen de prioridades

| # | Modificación | Dificultad | Impacto |
|---|---|---|---|
| 1 | dtype timesteps | 🟢 Fácil | 🟡 Bajo (buena práctica) |
| 2 | Documentar varianza | 🟢 Fácil | 🟢 Claridad |
| 3 | num_action_nodes | 🟢 Fácil | 🔴 Alto |
| 4 | Eliminar comentario | 🟢 Trivial | 🟢 Claridad |
| 5 | Split train/test | 🟡 Medio | 🔴 Crítico |
| 6 | Validation loss | 🟡 Medio | 🔴 Crítico |
| 7 | Checkpointing | 🟡 Medio | 🟡 Importante |
| 8 | Métricas evaluación | 🟡 Medio | 🔴 Crítico |
| 9 | Visualización | 🟢 Fácil | 🟡 Importante |

> [!TIP]
> Recomiendo empezar por **#1, #2, #3, #4** (correcciones rápidas), luego **#5 y #6** (separación de datos + validation), después **#7** (checkpointing), y finalmente **#8 y #9** (evaluación y visualización).

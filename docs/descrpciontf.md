Conversación con Claude

Edited pyproject.toml
Edited pyproject.toml

Perfecto. Veo que ya agregaste `torch-geometric` al `pyproject.toml`. Te explico celda por celda con todo el detalle pedido.

---

## Celda 1 — Imports y verificación de dependencias

```python
import h5py
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors

print(f"PyTorch: {torch.__version__}")
import torch_geometric
print(f"PyTorch Geometric: {torch_geometric.__version__}")
```

### Línea por línea

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `import h5py` | Carga la librería para leer archivos HDF5 (tu dataset). HDF5 es un formato de archivo jerárquico pensado para grandes volúmenes de datos numéricos. | Abre el canal de lectura hacia `dataset_instant_policy_2d.h5` |
| `import numpy as np` | NumPy es tu librería de álgebra lineal. La necesitás para manipular los arrays de puntos XY antes de convertirlos a tensores de PyTorch. | Operaciones con matrices numéricas: `np.arange`, `np.stack`, etc. |
| `import torch` | Importa PyTorch, el framework de deep learning. Todo tensor, modelo y operación diferenciable vivirá aquí. | Motor de cómputo: tensores, autograd, GPU/CPU |
| `from torch_geometric.data import HeteroData` | Importa la clase central de PyTorch Geometric (PyG) que representa un **grafo heterogéneo**. Un `HeteroData` es un diccionario inteligente que agrupa nodos y aristas de distintos tipos bajo un mismo objeto. | Es equivalente a un `dict` con acceso por tipo: `data['context']`, `data['action', 'to', 'context']` |
| `from sklearn.neighbors import NearestNeighbors` | Importa el algoritmo de k-Nearest Neighbors de scikit-learn. Lo usaremos para encontrar, dado un punto de acción, cuáles son los `k` puntos de contexto más cercanos en el espacio 2D. | Implementa un árbol de búsqueda espacial (ball tree / KD-tree) para consultas de vecindad |
| `print(f"PyTorch: {torch.__version__}")` | Verificación de entorno: imprime la versión de torch instalada. | Diagnóstico: confirma que la instalación es correcta antes de empezar |
| `import torch_geometric` + `print(...)` | Igual que antes pero para PyG. Es buena práctica verificar versiones porque PyG es muy sensible a la versión de torch subyacente. | Diagnóstico de compatibilidad entre torch y torch-geometric |

---

## Celda 2 — Función de normalización

```python
def normalize_pair(context: np.ndarray, target: np.ndarray):
    combined = np.concatenate([context, target], axis=0)
    min_vals = combined.min(axis=0)
    max_vals = combined.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    context_norm = 2.0 * (context - min_vals) / range_vals - 1.0
    target_norm  = 2.0 * (target  - min_vals) / range_vals - 1.0
    return context_norm, target_norm
```

### ¿Por qué normalizar el par *conjuntamente*?

Si normalizaras cada trayectoria por separado, destruirías la relación espacial entre ellas. Por ejemplo: si el contexto va de 0 a 10 y el target va de 0 a 1, ambos quedarían mapeados a [-1, 1], borrando el hecho de que el target era 10 veces más pequeño. Al normalizarlos juntos, se preserva esa geometría relativa, que es exactamente lo que la GNN necesita para que el grafo tenga sentido.

### Línea por línea

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `combined = np.concatenate([context, target], axis=0)` | Apila verticalmente los dos arrays de puntos en uno solo. Si cada uno tiene shape `(50, 2)`, el resultado es `(100, 2)`. Es como poner las dos trayectorias en una "tabla" única para calcular estadísticas globales. | `axis=0` = apila filas (puntos), no columnas (coordenadas) |
| `min_vals = combined.min(axis=0)` | Calcula el mínimo de todo el conjunto de 100 puntos, **por columna**. El resultado es un vector `[min_x, min_y]` que representa el punto más a la izquierda y más abajo de toda la escena. | `axis=0` = reduce a lo largo de los puntos, resultado: shape `(2,)` |
| `max_vals = combined.max(axis=0)` | Análogo al anterior: el punto más a la derecha y más arriba. | Resultado: shape `(2,)` |
| `range_vals = max_vals - min_vals` | Calcula el rango de variación en cada dimensión: `[rango_x, rango_y]`. Es el "tamaño" de la escena en X y en Y. | Si los puntos van de -2 a 3 en X, `range_vals[0] = 5` |
| `range_vals[range_vals == 0] = 1.0` | Caso borde: si todos los puntos tienen la misma coordenada en un eje (trayectoria horizontal o vertical pura), el rango sería 0 y ocurriría una división por cero. Se reemplaza por 1 para dejarlo sin cambio. | Guard clause contra `ZeroDivisionError` |
| `context_norm = 2.0 * (context - min_vals) / range_vals - 1.0` | Aplica la fórmula de normalización min-max al rango [-1, 1]: `y = 2*(x - min)/(max - min) - 1`. Con los stats globales calculados sobre el par combinado. | Primero `(x - min)/range` lleva al rango [0, 1]; luego `* 2 - 1` lleva a [-1, 1] |
| `target_norm = 2.0 * (target - min_vals) / range_vals - 1.0` | Exactamente lo mismo, pero para los puntos del target. Usa los **mismos** `min_vals` y `range_vals` del par combinado, preservando la relación espacial. | Si ambas trayectorias son iguales, `context_norm == target_norm` |
| `return context_norm, target_norm` | Devuelve ambas trayectorias normalizadas como una tupla. Python asigna automáticamente: `ctx_n, tgt_n = normalize_pair(ctx, tgt)`. | Retorno múltiple (tuple unpacking) |

---

## Celda 3 — Función principal: `build_hetero_graph`

Esta es la función central. La desgloso en 4 bloques.

```python
def build_hetero_graph(
    context: np.ndarray,
    target: np.ndarray,
    k_neighbors: int = 5,
    normalize: bool = True,
) -> HeteroData:
```

### Firma de la función

| Parámetro | Tipo | Significado |
|---|---|---|
| `context` | `np.ndarray (50, 2)` | Trayectoria de ejemplo (los "puntos rojos") |
| `target` | `np.ndarray (50, 2)` | Trayectoria que el modelo predecirá |
| `k_neighbors` | `int` | Cuántos nodos de contexto conectar a cada nodo de acción (k-NN) |
| `normalize` | `bool` | Si se normaliza el par antes de construir el grafo |
| `-> HeteroData` | type hint | La función siempre devuelve un objeto `HeteroData` |

### Bloque A — Normalización y creación del objeto

```python
    if normalize:
        context, target = normalize_pair(context, target)

    data = HeteroData()
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `if normalize:` | Switch opcional. Permite llamar a la función con datos ya normalizados sin doble normalización. | `normalize=False` → se omite el paso de escala |
| `context, target = normalize_pair(...)` | Reemplaza los arrays originales por sus versiones normalizadas al rango [-1, 1]. A partir de aquí, todo se opera sobre datos normalizados. | Tuple unpacking: Python desempaqueta el retorno de `normalize_pair` |
| `data = HeteroData()` | Crea un grafo heterogéneo vacío. Es como instanciar un diccionario anidado especializado. Todavía no tiene nodos ni aristas; se los iremos agregando como atributos. | Equivale a `data = {}` pero con semántica de grafo y métodos de PyG |

### Bloque B — Definición de nodos

```python
    data['context'].x = torch.tensor(context, dtype=torch.float32)
    data['action'].x  = torch.tensor(target,  dtype=torch.float32)

    n_action  = target.shape[0]
    n_context = context.shape[0]
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `data['context'].x = ...` | Define el tipo de nodo `'context'` y le asigna su **matriz de features**. En PyG, `.x` es siempre la matriz de características de los nodos, con shape `(num_nodos, num_features)`. | Crea una entrada en el diccionario interno de `data` bajo la clave `'context'` |
| `torch.tensor(context, dtype=torch.float32)` | Convierte el array NumPy `(50, 2)` a un tensor de PyTorch de punto flotante de 32 bits. Los modelos de PyTorch operan sobre tensores, no sobre arrays NumPy. `float32` es el estándar en deep learning por balance entre precisión y memoria. | `np.array → torch.Tensor`. Shape se preserva: `(50, 2)` |
| `data['action'].x = ...` | Análogo: define el tipo de nodo `'action'` con los puntos del target. Estos serán los nodos que la GNN modificará iterativamente durante el proceso de difusión. | Misma shape: `(50, 2)` |
| `n_action = target.shape[0]` | Extrae el número de nodos de acción (50). Se usará para construir los índices de aristas sin hardcodear valores. | `target.shape` devuelve la tupla `(50, 2)`; `[0]` extrae el 50 |
| `n_context = context.shape[0]` | Análogo para los nodos de contexto (también 50 en tu caso, pero pueden diferir en general). | Idem |

### Bloque C — Aristas `action → context` (k-NN)

```python
    k = min(k_neighbors, n_context)
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    knn.fit(context)
    _, indices = knn.kneighbors(target)

    src = np.repeat(np.arange(n_action), k)
    dst = indices.flatten()

    data['action', 'to', 'context'].edge_index = torch.tensor(
        np.stack([src, dst], axis=0), dtype=torch.long
    )
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `k = min(k_neighbors, n_context)` | Capping de seguridad: si pedís más vecinos que nodos de contexto existentes, k-NN lanza error. Se limita k al máximo posible. | Previene `ValueError` en datasets pequeños |
| `knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')` | Instancia el buscador de vecinos. `ball_tree` es un algoritmo de partición espacial eficiente para datos de baja dimensión (2D en tu caso). Construye una estructura de árbol que permite búsquedas O(log n) en lugar de O(n). | Analogía: es como crear un índice espacial (similar a un R-tree en bases de datos geoespaciales) |
| `knn.fit(context)` | Le dice al árbol "estos son los puntos donde buscar". El árbol se construye sobre los **nodos de contexto**: cuando queramos el vecino más cercano a un nodo de acción, lo buscará dentro de este conjunto. | Fase de indexación. Después de esto, `knn` sabe dónde están los 50 puntos de contexto |
| `_, indices = knn.kneighbors(target)` | Para cada uno de los 50 puntos de **target** (acción), encuentra los `k` puntos de **contexto** más cercanos. Devuelve distancias (ignoradas con `_`) e **índices** de los vecinos. `indices` tiene shape `(50, k)`: fila i = los k vecinos del nodo de acción i. | `kneighbors` usa la métrica Euclidiana por defecto: $d = \sqrt{\Delta x^2 + \Delta y^2}$ |
| `src = np.repeat(np.arange(n_action), k)` | Crea el array de **nodos fuente** (action). `np.arange(50)` = `[0,1,...,49]`. `np.repeat(..., k)` repite cada elemento k veces: `[0,0,0,0,0, 1,1,1,1,1, ..., 49,49,49,49,49]` si k=5. Cada nodo de acción aparece k veces porque tiene k aristas salientes. | Shape final: `(n_action * k,)` = `(250,)` |
| `dst = indices.flatten()` | Aplana la matriz de índices `(50, k)` a un vector `(250,)`. Cada elemento es el índice del nodo de contexto destino. Se alinea perfectamente con `src`: el par `(src[i], dst[i])` define la i-ésima arista. | `[[v0, v1, ..., vk], ...]` → `[v0, v1, ..., vk, v0', v1', ...]` |
| `np.stack([src, dst], axis=0)` | Apila `src` y `dst` como filas para crear una matriz `(2, 250)`. Esta es la convención que exige PyG para `edge_index`: la primera fila son los nodos fuente, la segunda los nodos destino. | **Formato PyG**: `edge_index[0]` = sources, `edge_index[1]` = destinations |
| `dtype=torch.long` | Las aristas deben ser índices enteros (int64). `torch.long` es el tipo estándar de PyG para `edge_index`. | Si usaras `float32` en los índices, PyG lanzaría un error en la propagación de mensajes |
| `data['action', 'to', 'context'].edge_index = ...` | La clave del tipo de arista es una **tripla** `(tipo_src, relación, tipo_dst)`. PyG usa esta convención para saber en qué nodos buscar al propagar mensajes a través de esta arista. | Esta arista habilita el mensaje: "cada nodo acción puede 'preguntar' a sus k nodos contexto vecinos" |

### Bloque D — Aristas `action → action` (secuencial)

```python
    src_seq = np.arange(n_action - 1)
    dst_seq = np.arange(1, n_action)

    data['action', 'to', 'action'].edge_index = torch.tensor(
        np.stack([src_seq, dst_seq], axis=0), dtype=torch.long
    )

    return data
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `src_seq = np.arange(n_action - 1)` | Genera `[0, 1, 2, ..., 48]`: los índices de todos los nodos de acción *excepto el último*. Cada uno será el punto de **partida** de una arista secuencial. | `np.arange(49)` → array de 49 elementos |
| `dst_seq = np.arange(1, n_action)` | Genera `[1, 2, 3, ..., 49]`: los índices de todos los nodos de acción *excepto el primero*. Cada uno será el punto de **llegada**. El par `(src_seq[i], dst_seq[i])` forma la arista `i → i+1`. | `np.arange(1, 50)` → array de 49 elementos |
| `np.stack([src_seq, dst_seq], axis=0)` | Igual que antes: crea la matriz `(2, 49)` en formato PyG. | 49 aristas secuenciales: `0→1, 1→2, ..., 48→49` |
| `data['action', 'to', 'action'].edge_index = ...` | Registra las aristas intra-acción. Al propagar mensajes por estas aristas, cada nodo de acción recibe información de su vecino temporal anterior, lo que da **continuidad y suavidad** a la trayectoria generada. | Estructura análoga al "self-loop temporal" en modelos secuenciales |
| `return data` | Devuelve el objeto `HeteroData` completamente configurado con 2 tipos de nodos y 2 tipos de aristas. | El objeto es inmutable en cuanto a su topología; las features `.x` pueden cambiar durante el forward de la GNN |

---

## Celda 4 — Carga desde HDF5 e inspección

```python
def load_demo_as_graph(hdf5_path: str, demo_key: str, **kwargs) -> HeteroData:
    with h5py.File(hdf5_path, 'r') as f:
        context = f['data'][demo_key]['context'][:]
        target  = f['data'][demo_key]['target'][:]
    return build_hetero_graph(context, target, **kwargs)

graph = load_demo_as_graph(HDF5_PATH, 'demo_0', k_neighbors=5)
print(graph)
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `**kwargs` en la firma | Permite pasar argumentos opcionales (`k_neighbors`, `normalize`) directamente a `build_hetero_graph` sin reescribirlos. El `**` desempaqueta el diccionario de argumentos nombrados. | `load_demo_as_graph(..., k_neighbors=5)` → `build_hetero_graph(..., k_neighbors=5)` |
| `with h5py.File(hdf5_path, 'r') as f:` | Abre el archivo HDF5 en modo sólo lectura (`'r'`). El bloque `with` garantiza que el archivo se cierre correctamente al terminar, incluso si ocurre una excepción. | Equivale al patrón `f = open(...); try: ...; finally: f.close()` |
| `f['data'][demo_key]['context'][:]` | Navega la jerarquía del HDF5: `data` → `demo_0` → `context`. El `[:]` es un slice que carga **todos los datos** del dataset HDF5 en memoria como un NumPy array. Sin `[:]`, `f['data']['demo_0']['context']` es solo un *handle* (referencia perezosa), no los datos. | `[:]` = "dame todo el contenido como array". Shape resultante: `(50, 2)` |
| `print(graph)` | PyG tiene un `__repr__` que muestra el resumen del grafo: tipos de nodos con sus shapes y tipos de aristas. Útil para verificar que la construcción fue correcta. | Output esperado: `HeteroData(context={x=[50,2]}, action={x=[50,2]}, ...)` |

---

## Celda 5 — Visualización del grafo

```python
def plot_hetero_graph(graph: HeteroData, max_edges_shown: int = 30):
    ctx     = graph['context'].x.numpy()
    act     = graph['action'].x.numpy()
    edge_ac = graph['action', 'to', 'context'].edge_index.numpy()
    edge_aa = graph['action', 'to', 'action'].edge_index.numpy()
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `.x.numpy()` | Convierte el tensor de PyTorch de vuelta a NumPy para que Matplotlib pueda graficarlo. PyTorch y NumPy comparten memoria cuando el tensor está en CPU, así que la conversión es gratuita. | `torch.Tensor → np.ndarray`. Requiere que el tensor esté en CPU (no GPU) |
| `edge_ac = graph['action', 'to', 'context'].edge_index.numpy()` | Recupera la matriz de aristas `(2, 250)` como NumPy. Fila 0 = índices de nodos acción (fuente), fila 1 = índices de nodos contexto (destino). | Shape: `(2, n_action * k)` |

```python
    for i in range(min(max_edges_shown, edge_ac.shape[1])):
        s, d = edge_ac[0, i], edge_ac[1, i]
        ax.plot([act[s, 0], ctx[d, 0]], [act[s, 1], ctx[d, 1]], ...)
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `min(max_edges_shown, edge_ac.shape[1])` | Limita las aristas graficadas a 30 para no saturar el plot (con k=5 habría 250). | Capping estético: el grafo completo sería visualmente ilegible |
| `s, d = edge_ac[0, i], edge_ac[1, i]` | Desempaqueta los índices de la i-ésima arista: `s` = nodo de acción fuente, `d` = nodo de contexto destino. | Acceso por fila y columna a la matriz |
| `ax.plot([act[s, 0], ctx[d, 0]], [act[s, 1], ctx[d, 1]])` | Dibuja un segmento de línea entre el punto de acción `s` y el punto de contexto `d`. El primer arg son las coordenadas X de los dos extremos, el segundo las Y. | Matplotlib conecta dos puntos: `(act[s,0], act[s,1])` → `(ctx[d,0], ctx[d,1])` |

```python
    for i in range(edge_aa.shape[1]):
        s, d = edge_aa[0, i], edge_aa[1, i]
        ax.plot([act[s, 0], act[d, 0]], [act[s, 1], act[d, 1]], ...)
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `edge_aa.shape[1]` | Número total de aristas secuenciales (49). Estas sí se grafican todas porque son pocas y forman la "cadena" de la trayectoria. | `shape[1]` = número de columnas = número de aristas |
| `ax.plot([act[s,0], act[d,0]], ...)` | Dibuja el segmento entre el punto `t` y el punto `t+1` de la trayectoria de acción. Si graficás todas estas aristas seguidas, obtenés la trayectoria como polilínea. | Visualiza la cadena secuencial de la trayectoria |

---

## Celda 6 — Generador del dataset completo

```python
def iter_dataset_as_graphs(hdf5_path: str, k_neighbors: int = 5):
    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted(f['data'].keys())
        for key in demo_keys:
            context = f['data'][key]['context'][:]
            target  = f['data'][key]['target'][:]
            yield key, build_hetero_graph(context, target, k_neighbors=k_neighbors)
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `def iter_dataset_as_graphs(...)` | Es un **generador** Python (lo delata el `yield`). A diferencia de una función que devuelve una lista, un generador produce los elementos **uno a la vez bajo demanda**, sin cargar todo el dataset en RAM. Con 1000 demos de grafos, esto puede ahorrar varios GB de memoria. | Lazy evaluation: el archivo HDF5 *permanece abierto* durante toda la iteración |
| `demo_keys = sorted(f['data'].keys())` | Obtiene todas las claves de los grupos dentro de `data/` y las ordena alfabéticamente. Sin `sorted`, el orden puede ser arbitrario (HDF5 no garantiza orden). Con `sorted`, el orden es `demo_0, demo_1, ..., demo_999`. | `f['data'].keys()` devuelve un objeto iterable, no una lista. `sorted()` lo materializa en lista ordenada |
| `for key in demo_keys:` | Itera sobre cada demo. El archivo HDF5 queda abierto durante todo el loop gracias al `with`. | Cada iteración lee solo una demo del disco |
| `yield key, build_hetero_graph(...)` | La palabra clave `yield` es lo que hace que sea un generador. Al llamar `next()` sobre el generador (implícito en el `for`), ejecuta hasta el próximo `yield`, devuelve el par `(key, HeteroData)`, y **pausa** la ejecución ahí hasta la siguiente llamada. | Equivale a `return` pero sin terminar la función: la retoma donde la dejó |

```python
for key, g in iter_dataset_as_graphs(HDF5_PATH):
    assert g['context'].x.shape == (50, 2)
    assert g['action'].x.shape  == (50, 2)
    count += 1
```

| Línea | Concepto | Equivalente programático |
|---|---|---|
| `for key, g in iter_dataset_as_graphs(...)` | Consume el generador. Python llama a `next()` en cada iteración, el generador lee la demo, construye el grafo y lo devuelve. | Tuple unpacking: `key = 'demo_0'`, `g = HeteroData(...)` |
| `assert g['context'].x.shape == (50, 2)` | Assertion de forma: verifica que los nodos de contexto tienen exactamente 50 filas y 2 columnas. Si alguna demo está corrupta o mal guardada, esto lanzaría `AssertionError` inmediatamente. | `assert condicion` → lanza `AssertionError` si `condicion` es `False` |
| `count += 1` | Contador simple para confirmar al final que se procesaron las 1000 demos. | `count` comienza en 0 antes del loop |
| `print(f"✅ {count} grafos generados correctamente")` | Confirmación visual. Si el output es `✅ 1000 grafos generados correctamente`, el pipeline de datos está listo para alimentar la GNN. | |

---

### Visión global del flujo de datos

```
HDF5 (demo_i)
  context (50,2) ──┐
  target  (50,2) ──┤
                   ▼
            normalize_pair()
                   │
                   ▼
          build_hetero_graph()
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
  data['context'].x    data['action'].x
      (50, 2)               (50, 2)
         │                    │
         └──── k-NN ──────────┤  → edge_index (2, 250)
                              │
                    secuencial t→t+1  → edge_index (2, 49)
                              │
                              ▼
                         HeteroData ──→ GNN (próximo paso)
```
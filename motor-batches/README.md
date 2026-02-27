# MiniTensor Rust Engine 🚀

**MiniTensor** es un motor de redes neuronales de alto rendimiento que fusiona la seguridad y velocidad de **Rust** con la flexibilidad de **Python**. Diseñado para ser ligero, persistente y optimizado para entornos de **Cloud Computing**.


## 🧠 Arquitectura Híbrida
A diferencia de los frameworks tradicionales, MiniTensor delega el 100% del cálculo tensorial a binarios compilados en Rust, eliminando el *Global Interpreter Lock (GIL)* de Python durante las fases de entrenamiento pesado.

* **Core (Rust):** Implementa Backpropagation y gradiente descendente estocástico usando `ndarray`.
* **Interface (Python):** Proporciona una API limpia y expresiva para investigación y pruebas rápidas.
* **Persistencia:** Modelos serializados en JSON de alta densidad (~471 bytes), ideales para arquitecturas Edge y microservicios.

## 📊 Benchmarks de Rendimiento
Optimización mediante procesamiento por lotes (Batches) con vectorización **SIMD**:

| Batch Size | Tiempo (ms) | Samples/sec | Eficiencia |
| :--- | :--- | :--- | :--- |
| 1 | 5.98 | 166 | Base |
| 256 | 88.11 | 2,905 | **+17.5x** |



## 🛠️ Stack Tecnológico
* **Rust (Kernel):** `ndarray` para álgebra lineal, `PyO3` para bindings de Python.
* **Serde:** Serialización ultra-eficiente para modelos portátiles.
* **Docker:** Entorno de compilación multi-etapa para despliegues *zero-config*.

## 🚀 Despliegue Rápido (Docker)
Construye y ejecuta el motor en cualquier entorno de nube en segundos:

```bash
# Construir la imagen
docker build -t motor-ia .

# Ejecutar entrenamiento y predicción (XOR Test)
docker run --rm motor-ia python3 -c "import motor_batches as nb; import numpy as np; red = nb.RedBatched([2, 4, 1], 0.1); X = np.array([[0.,1.],[1.,0.],[1.,1.],[0.,0.]]); y = np.array([[1.],[1.],[0.],[0.]]); [red.train_batch(X, y) for _ in range(2000)]; print(f'\n--- IA PREDICIENDO ---\nEntrada [0,1] -> Prediccion: {red.predict(np.array([[0.0, 1.0]]))[0][0]:.4f}')"

```

## 📖 Guía de Ingeniería

### 1. Inicialización y Entrenamiento

El motor permite definir arquitecturas multicapa de forma dinámica.

```python
import motor_batches as nb
import numpy as np

# Arquitectura: [Entrada, Capas Ocultas..., Salida]
red = nb.RedBatched([2, 4, 1], lr=0.1)

# Datos de entrenamiento (Lógica XOR)
X = np.array([[0.,1.], [1.,0.], [1.,1.], [0.,0.]])
y = np.array([[1.], [1.], [0.], [0.]])

# Ciclo de entrenamiento optimizado en Rust
for _ in range(2000):
    loss = red.train_batch(X, y)

```

### 2. Inferencia y Predicción

Una vez entrenado, el motor alcanza una precisión superior al **99%** en problemas no lineales.

```python
# Inferencia de alta velocidad
prediccion = red.predict(np.array([[0.0, 1.0]]))
print(f"Resultado: {prediccion[0][0]:.4f}") # Output esperado: ~0.9911

```

### 3. Persistencia de Modelo

El "cerebro" de la red se guarda en un archivo JSON agnóstico que puede ser cargado en cualquier sistema sin re-entrenar.

```python
red.save("modelo_ia_rust.json")
nueva_red = nb.RedBatched.load("modelo_ia_rust.json")

```

---

Desarrollado por **asoleal** como parte de la exploración en IA de alto rendimiento con Rust.

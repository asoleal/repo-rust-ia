# Motor-Rust-V3: IA de Alto Rendimiento

Motor de redes neuronales profunda (MLP) desarrollado desde cero en **Rust**, optimizado para el hardware de la Inspiron 3593 (8 hilos).

## 🚀 Características Técnicas
- **Arquitectura:** Perceptrón Multicapa (MLP) modular.
- **Optimizador:** Adam (Adaptive Moment Estimation) con corrección de sesgo.
- **Funciones de Activación:** LeakyReLU (ocultas) y Softmax (salida).
- **Dataset:** MNIST (60,000 imágenes de entrenamiento, 10,000 de test).
- **Rendimiento:** ~45,000 imágenes/segundo en entrenamiento.
- **Precisión:** >93% Accuracy en solo 10 épocas.

## 🛠️ Estructura del Proyecto
- `src/lib.rs`: Núcleo del motor, backpropagation y lógica de Adam.
- `src/mnist_loader.rs`: Cargador binario de alto rendimiento para archivos IDX.
- `src/main.rs`: Pipeline de entrenamiento con Mini-batches y Shuffling.
- `src/bin/inferencia.rs`: Herramienta de predicción con visualización ASCII en terminal.
- `src/bin/benchmark.rs`: Test de estrés paralelo para validación de CPU.

## 📦 Uso
1. **Entrenar:** `cargo run --release --bin motor-rust-v3`
2. **Predecir:** `cargo run --release --bin inferencia -- [índice]`

## 📊 Resultados Recientes
| Época | Accuracy | Confianza Promedio |
|-------|----------|--------------------|
| 0     | 92.17%   | Alta               |
| 9     | 93.60%   | 99.99% (Softmax)   |

---
**Desarrollado en Rust para máxima eficiencia y seguridad de memoria.**

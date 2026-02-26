# Motor de IA en Rust: Redes Neuronales Dinámicas con Diagnóstico de Salud

Este repositorio contiene la evolución de un motor de aprendizaje profundo desarrollado desde cero utilizando **Rust** para el núcleo de cómputo y **Python** para la orquestación y visualización. El diseño se inspira en la arquitectura de tres capas (Python-PyO3-Rust) destacada en investigaciones recientes como *MiniTensor* (Sarkar, 2026).

## 🚀 Características Principales

* **Núcleo de Alto Rendimiento:** Implementación de álgebra lineal y backpropagation manual optimizada con `ndarray` en Rust.
* **Diagnóstico Científico:** Monitoreo en tiempo real del ratio de neuronas ReLU vivas para detectar el gradiente desvaneciente (Vanishing Gradient).
* **Interoperabilidad Zero-Copy:** Integración fluida con NumPy a través de PyO3, minimizando la latencia de memoria entre lenguajes.
* **Arquitectura Auditable:** Un codebase minimalista diseñado para la investigación y educación, evitando el "bloat" de los frameworks comerciales.

## 🏗️ Estructura del Proyecto

El repositorio está organizado para mostrar la progresión técnica del desarrollo:

1.  **`back-propagation/`**: Implementación base de la retropropagación manual y capas densas.
2.  **`red-diagnosticada/`**: Versión avanzada que incorpora la clase `TrainingStats` para el análisis de salud del modelo.

## 📊 Análisis de Estabilidad

A diferencia de los frameworks de "caja negra", este motor expone métricas internas de salud. Durante el entrenamiento, el motor reporta:
* **MSE Loss:** Calculado de forma nativa en Rust para mayor precisión.
* **Alive Neurons Ratio:** Un indicador crítico de la estabilidad de la función de activación ReLU.



## 🛠️ Requisitos e Instalación

### Requisitos
* Rust (Edición 2021 o superior)
* Python 3.12+ (Optimizado para 3.14)
* Maturin

### Instalación
1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/asoleal/motor-ia-rust-python.git](https://github.com/asoleal/motor-ia-rust-python.git)
   cd motor-ia-rust-python/red-diagnosticada

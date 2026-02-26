# MiniTensor Rust Engine 🚀

Motor de redes neuronales de alto rendimiento desarrollado en **Rust** con interfaz de **Python**. Diseñado para ser ligero, persistente y listo para **Cloud Computing**.

## 📊 Benchmarks de Rendimiento
Optimización mediante procesamiento por lotes (Batches):

| Batch Size | Tiempo (ms) | Samples/sec |
|------------|-------------|-------------|
| 1          | 5.98        | 166         |
| 256        | 88.11       | 2,905       |

*Incremento de eficiencia: ~17.5x gracias a la vectorización SIMD en Rust.*

## 🛠️ Tecnologías
* **Rust**: Núcleo de cálculo (ndarray, PyO3).
* **Python**: Orquestación y pruebas.
* **Serde**: Serialización JSON ultra-ligera (~460 bytes por modelo).
* **Docker**: Despliegue agnóstico a la plataforma.

## 🚀 Ejecución en Cloud (Docker)
Para correr este proyecto sin configurar dependencias:
\`\`\`bash
docker build -t asoleal/motor-rust-ia:v1 .
docker run --rm asoleal/motor-rust-ia:v1
\`\`\`

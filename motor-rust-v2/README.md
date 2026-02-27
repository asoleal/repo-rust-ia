# Motor Rust IA Nativo (V2) 🦀⚡

Motor de redes neuronales ultra-veloz con backpropagation nativo implementado en Rust.

## Benchmarks (Core i7 / Inspiron 3593)
- **Batch Size:** 100 ejemplos.
- **Tiempo por época:** 13 microsegundos.
- **Entrenamiento total (1k épocas):** ~13.5 milisegundos.

## Características
- **Zero-Copy Training:** El bucle de entrenamiento ocurre enteramente en Rust.
- **Activación:** ReLU en capas ocultas y Sigmoide en la salida.
- **Persistencia:** Guardado/Carga de modelos en formato JSON.
- **Cloud Ready:** Configuración para GitHub Codespaces incluida.

## Ejecución
```bash
cargo run --release
```

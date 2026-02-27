# Motor Rust Native IA 🦀

Un motor de redes neuronales de alto rendimiento escrito 100% en Rust.

## Rendimiento
- **Inferencia:** ~1-2 microsegundos.
- **Entrenamiento (Backprop):** ~2 microsegundos por época.
- **Eficiencia:** Implementación nativa utilizando `ndarray`.

## Uso
Para ejecutar el benchmark de entrenamiento:
```bash
cargo run --release
```

# Motor Rust V7 - Industrial Edition 🏭

Este proyecto ha evolucionado de un prototipo a una **Librería de Inferencia Dinámica**.

## 🧬 Avances de la V7
- **Arquitectura**: CNN 1D con 12 filtros especializados en eliminación de ruido.
- **Despliegue**: Compila como `cdylib`, permitiendo integración nativa con Python, C++ y sistemas SCADA.
- **Visualización**: Sistema de exportación de activaciones para auditoría de decisiones de la IA.

## 🛠️ Uso como Librería
1. Compilar: `cargo build --release`
2. Localizar: `target/release/libmotor_rust_v7.so` (Linux)
3. Cargar desde Python usando `ctypes`.

## 📊 Rendimiento
Confirmado 100% de precisión en el Super Test bajo condiciones de ruido Gaussiano.

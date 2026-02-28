# Motor Rust V6: CNN de Monitoreo Industrial 🛡️

Sistema de detección de anomalías en motores mediante Redes Convolucionales 1D (CNN) desarrollado en **Rust puro**.

## 🚀 Características Destacadas
- **Precisión:** 100.00% bajo condiciones de ruido blanco gaussiano y señales fantasma.
- **Rendimiento:** Inferencia ultra-rápida (~100 nanosegundos por muestra).
- **Arquitectura:** Conv1D (12 filtros) + MaxPool1D + Dense, optimizada en **f32**.
- **Eficiencia:** Entrenamiento de 1,000,000 de operaciones en < 60 segundos.

## 🛠️ Estructura del Proyecto
- `src/nn/`: Implementación manual de capas neuronales (sin frameworks externos).
- `src/data/`: Generador de señales sintéticas con inyección de ruido hostil.
- `src/bin/stream.rs`: Simulador de monitoreo en tiempo real a 100Hz.
- `src/bin/super_test_v6.rs`: Script de validación masiva con 2,000 muestras.

## 📈 Resultados V6
| Métrica | Valor |
|---------|-------|
| Falsos Positivos | 0 |
| Falsos Negativos | 0 |
| Tiempo Entrenamiento | ~55s (10k samples) |
| Latencia Inferencia | < 0.15µs |


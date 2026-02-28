# 🦀 Motor Rust V4: Deep CNN from Scratch

Un motor de visión artificial construido en **Rust** desde cero, sin frameworks de alto nivel (como PyTorch o TensorFlow). Utiliza una arquitectura de Red Neuronal Convolucional (CNN) de dos capas para clasificar dígitos del dataset MNIST.

## 🚀 Características
* **Arquitectura Deep:** 2 capas de convolución, Max Pooling, Flatten y capa Densa.
* **Alto Rendimiento:** Procesamiento paralelo de convoluciones usando `Rayon`.
* **Persistencia:** Guardado y carga de pesos en formato JSON.
* **Precisión:** **91.5% Accuracy** en el set de prueba de 10,000 imágenes.

## 🛠️ Arquitectura de la Red
1. **Conv2D**: 16 filtros (3x3), activación LeakyReLU.
2. **MaxPool**: Reducción 2x2.
3. **Conv2D**: 16 filtros (3x3), activación LeakyReLU.
4. **MaxPool**: Reducción 2x2.
5. **Flatten**: Conversión a vector.
6. **Densa**: 400 entradas -> 10 salidas (Sigmoide).

## 📊 Uso
### Entrenamiento
```bash
cargo run --release --bin motor-rust-v4
```

### Evaluación (Arena)
```bash
cargo run --release --bin arena
```

### Inferencia Externa
Puedes probar tus propias imágenes (28x28, blanco sobre negro):
```bash
cargo run --release --bin ver_imagen -- tu_dibujo.png
```

## 📦 Requisitos
* Tener el dataset MNIST en la carpeta `/data`.
* Rust 1.70+

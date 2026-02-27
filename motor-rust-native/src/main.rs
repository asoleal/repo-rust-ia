use motor_rust_native::RedNativa;
use ndarray::prelude::*;
use std::time::Instant;

fn main() {
    println!("🚀 Iniciando Entrenamiento Nativo en Rust...");

    // 1. Crear datos sintéticos (XOR de ejemplo o Iris)
    // 4 entradas, 3 salidas (clases)
    let x = array![[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4]];
    let y = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]; // One-hot encoding

    // 2. Inicializar red: 4 entradas -> 8 ocultas -> 3 salidas
    let mut red = RedNativa::new_rust(vec![4, 8, 3], 0.1);

    // 3. Bucle de entrenamiento intenso (10,000 épocas)
    let ahora = Instant::now();
    for epoca in 0..10000 {
        let _loss = red.train_native(&x, &y);
        if epoca % 1000 == 0 {
            println!("Época {}: Loss calculada", epoca);
        }
    }
    let duracion = ahora.elapsed();

    println!("\n✅ Entrenamiento completado en: {:?}", duracion);
    println!("⚡ Tiempo promedio por paso de backprop: {:?} microsegundos", duracion.as_micros() / 10000);

    // 4. Probar predicción
    let test = array![[5.1, 3.5, 1.4, 0.2]];
    let prediccion = red.predict_native(&test);
    println!("\n📊 Test de predicción: {:?}", prediccion);
}

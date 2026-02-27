use motor_batches::RedBatched;
use ndarray::Array2;
use std::time::Instant;

fn main() {
    let path = "models/iris_v1.json";
    println!("--- 🦀 Rustia Native Inference ---");

    // 1. Cargar el modelo entrenado
    match RedBatched::load(path) {
        Ok(red) => {
            println!("✅ Modelo cargado correctamente.");

            // 2. Crear una entrada de prueba (Medidas típicas de una flor Iris)
            // Supongamos que estas son las medidas después de pasar por el StandardScaler
            let entrada = Array2::from_shape_vec((1, 4), vec![0.5, -0.2, 0.8, 1.2]).unwrap();

            // 3. Medir el tiempo de inferencia (Solo el cálculo matemático)
            let ahora = Instant::now();
            let prediccion = red.predict_pure(&entrada);
            let duracion = ahora.elapsed();

            // 4. Mostrar resultados
            println!("\n📊 Resultado de la Predicción:");
            println!("{:?}", prediccion);
            
            let clase = prediccion.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap();

            println!("🍀 Clase identificada: {}", clase);
            println!("⚡ Tiempo de ejecución: {} microsegundos", duracion.as_micros());
        }
        Err(e) => {
            eprintln!("❌ Error al cargar el modelo: {}", e);
            println!("Tip: Asegúrate de que 'models/iris_v1.json' exista.");
        }
    }
}

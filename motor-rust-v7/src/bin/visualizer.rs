use motor_rust_v7::data::generate_sensor_data;
use motor_rust_v7::ModelV7;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🧪 Generando datos de visualización V7...");
    
    // 1. Generar una muestra real de prueba
    let (x_raw, _) = generate_sensor_data(1, 128);
    let input = x_raw.mapv(|v| v as f32);
    
    // 2. Cargar el modelo entrenado
    let f = File::open("modelo_v6.json").expect("Error: No se encontró modelo_v6.json");
    let mut model: ModelV7 = serde_json::from_reader(f).unwrap();
    
    // 3. Obtener activaciones de la capa Conv1D
    let activation = model.conv.forward(&input);
    
    // 4. Preparar JSON (Aplanando los arrays para compatibilidad)
    let output_data = serde_json::json!({
        "input": input.iter().cloned().collect::<Vec<f32>>(),
        "activations": activation.iter().cloned().collect::<Vec<f32>>(),
        "shape": activation.dim(),
        "filters": 12
    });

    let mut file = File::create("visualizacion.json").unwrap();
    file.write_all(output_data.to_string().as_bytes()).unwrap();
    
    println!("✅ Archivo 'visualizacion.json' generado.");
    println!("📊 Dimensiones de la activación: {:?}", activation.dim());
}

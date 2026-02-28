use motor_rust_v5::data::generate_sensor_data;
use motor_rust_v5::nn::conv1d::Conv1D;
use motor_rust_v5::nn::ops::{leaky_relu, MaxPool1D};
use motor_rust_v5::nn::dense::Dense;
use ndarray::Axis;
use serde::{Serialize, Deserialize};
use std::fs::File;

#[derive(Serialize, Deserialize)]
struct ModelV5 { conv: Conv1D, dense: Dense }

fn main() {
    println!("--- 🛡️ Motor V5: Monitor de Estado (Inferencia Real) ---");
    
    // Cargar el modelo entrenado
    let f = File::open("modelo_v5.json").expect("¡Primero debes entrenar con el binario motor-rust-v5!");
    let mut model: ModelV5 = serde_json::from_reader(f).unwrap();
    let mut pool = MaxPool1D::new(2);

    // Generar 50 muestras de prueba nuevas
    let (x_test, y_test) = generate_sensor_data(50, 128);
    let mut aciertos = 0;

    println!("{:<10} | {:<10} | {:<10} | {}", "Muestra", "Real", "Predicho", "Estado");
    println!("{:-<50}", "");

    for i in 0..50 {
        let input = x_test.index_axis(Axis(0), i).insert_axis(Axis(0)).to_owned();
        let target = y_test.index_axis(Axis(0), i);

        // --- FORWARD PASS (INFERENCIA) ---
        let z1 = model.conv.forward(&input);
        let a1 = leaky_relu(&z1);
        let p1 = pool.forward(&a1);
        
        // Medimos el tamaño antes de consumir p1
        let total_elements = p1.len();
        let flat = p1.into_shape((1, total_elements)).unwrap();
        let output = model.dense.forward(&flat);

        // Argmax: Clase 0 (Normal) o Clase 1 (Falla)
        let pred = if output[[0, 1]] > output[[0, 0]] { 1 } else { 0 };
        let real = if target[1] > target[0] { 1 } else { 0 };
        
        if pred == real { aciertos += 1; }
        
        if i < 15 { // Mostrar solo las primeras 15 para no saturar la terminal
            let real_str = if real == 1 { "FALLA" } else { "NORMAL" };
            let pred_str = if pred == 1 { "FALLA" } else { "NORMAL" };
            let icon = if pred == real { "✅" } else { "❌" };
            println!("{:<10} | {:<10} | {:<10} | {}", i + 1, real_str, pred_str, icon);
        }
    }

    println!("{:-<50}", "");
    println!("PRECISIÓN FINAL: {}/50 ({:.1}%)", aciertos, (aciertos as f32 / 50.0) * 100.0);
}

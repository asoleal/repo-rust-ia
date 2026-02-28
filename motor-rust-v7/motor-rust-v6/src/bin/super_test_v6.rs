use motor_rust_v6::data::generate_sensor_data;
use motor_rust_v6::nn::conv1d::Conv1D;
use motor_rust_v6::nn::ops::{leaky_relu, MaxPool1D};
use motor_rust_v6::nn::dense::Dense;
use ndarray::Axis;
use serde::{Serialize, Deserialize};
use std::fs::File;

#[derive(Serialize, Deserialize)]
struct ModelV6 { conv: Conv1D, dense: Dense }

fn main() {
    println!("--- 🛡️ SUPER TEST V6: EVALUACIÓN BAJO RUIDO ---");
    let f = File::open("modelo_v6.json").expect("No se encontró el modelo V6");
    let mut model: ModelV6 = serde_json::from_reader(f).unwrap();
    let pool = MaxPool1D::new(2);

    let num_pruebas = 2000;
    let (x_raw, y_raw) = generate_sensor_data(num_pruebas, 128);
    let x_test = x_raw.mapv(|v| v as f32);
    let y_test = y_raw.mapv(|v| v as f32);

    let (mut vp, mut vn, mut fp, mut fn_error) = (0, 0, 0, 0);

    for i in 0..num_pruebas {
        let input = x_test.index_axis(Axis(0), i).insert_axis(Axis(0)).to_owned();
        let target = y_test.index_axis(Axis(0), i);

        let z1 = model.conv.forward(&input);
        let a1 = leaky_relu(&z1);
        let p1 = pool.forward(&a1);
        let flat = p1.into_shape((1, 12 * 62)).unwrap();
        let output = model.dense.forward(&flat);

        let pred = if output[[0, 1]] > output[[0, 0]] { 1 } else { 0 };
        let real = if target[1] > target[0] { 1 } else { 0 };

        match (real, pred) {
            (1, 1) => vp += 1,
            (0, 0) => vn += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_error += 1,
            _ => (),
        }
    }

    println!("\n📊 MATRIZ DE CONFUSIÓN (2000 MUESTRAS RUIDOSAS):");
    println!("-------------------------------------------");
    println!("✅ Normales Correctos:  {}", vn);
    println!("🔥 Fallas Detectadas:   {}", vp);
    println!("⚠️ Falsas Alarmas:      {}", fp);
    println!("💀 Fallas Omitidas:     {}", fn_error);
    println!("-------------------------------------------");
    println!("🎯 PRECISIÓN FINAL:     {:.2}%", (vp + vn) as f32 / num_pruebas as f32 * 100.0);
}

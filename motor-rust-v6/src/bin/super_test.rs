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
    println!("--- 🚀 MOTOR V5: PRUEBA DE ESTRÉS EXHAUSTIVA ---");
    let f = File::open("modelo_v5.json").expect("Debes tener el modelo entrenado.");
    let mut model: ModelV5 = serde_json::from_reader(f).unwrap();
    let mut pool = MaxPool1D::new(2);

    let num_pruebas = 1000;
    let (x_test, y_test) = generate_sensor_data(num_pruebas, 128);

    let mut verdaderos_positivos = 0; // Era falla y dijo falla
    let mut verdaderos_negativos = 0; // Era normal y dijo normal
    let mut falsos_positivos = 0;    // Era normal y dijo falla (Falsa alarma)
    let mut falsos_negativos = 0;    // Era falla y dijo normal (Peligro)

    for i in 0..num_pruebas {
        let input = x_test.index_axis(Axis(0), i).insert_axis(Axis(0)).to_owned();
        let target = y_test.index_axis(Axis(0), i);

        let z1 = model.conv.forward(&input);
        let a1 = leaky_relu(&z1);
        let p1 = pool.forward(&a1);
        let flat = p1.into_shape((1, p1.len())).unwrap();
        let output = model.dense.forward(&flat);

        let pred = if output[[0, 1]] > output[[0, 0]] { 1 } else { 0 };
        let real = if target[1] > target[0] { 1 } else { 0 };

        match (real, pred) {
            (1, 1) => verdaderos_positivos += 1,
            (0, 0) => verdaderos_negativos += 1,
            (0, 1) => falsos_positivos += 1,
            (1, 0) => falsos_negativos += 1,
            _ => (),
        }
    }

    let precision = (verdaderos_positivos + verdaderos_negativos) as f32 / num_pruebas as f32 * 100.0;

    println!("\n📊 RESULTADOS SOBRE {} MUESTRAS:", num_pruebas);
    println!("-------------------------------------------");
    println!("✅ Verdaderos Normales: {}", verdaderos_negativos);
    println!("🔥 Verdaderos Fallas:   {}", verdaderos_positivos);
    println!("⚠️ Falsos Positivos:    {} (Falsas Alarmas)", falsos_positivos);
    println!("💀 Falsos Negativos:    {} (Anomalías No Detectadas)", falsos_negativos);
    println!("-------------------------------------------");
    println!("🎯 PRECISIÓN GLOBAL:    {:.2}%", precision);
}

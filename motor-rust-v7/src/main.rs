mod data;
mod nn;

use data::generate_sensor_data;
use nn::conv1d::Conv1D;
use nn::ops::{leaky_relu, leaky_relu_prime, MaxPool1D};
use nn::dense::Dense;
use ndarray::{Axis, Array2};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::time::Instant;

#[derive(Serialize, Deserialize)]
struct ModelV6 { conv: Conv1D, dense: Dense }

fn main() {
    println!("--- 🛡️ Motor Rust V6: Stress & Hardening (10k samples) ---");
    let seq_len = 128;
    let num_filters = 12; // Más filtros para manejar el ruido
    let kernel_size = 5;

    let (x_raw, y_raw) = generate_sensor_data(10000, seq_len);
    let x_train = x_raw.mapv(|v| v as f32);
    let y_train = y_raw.mapv(|v| v as f32);

    let mut conv = Conv1D::new(num_filters, 1, kernel_size);
    let pool = MaxPool1D::new(2);
    let pool_len = ((seq_len - kernel_size) + 1) / 2;
    let mut dense = Dense::new(num_filters * pool_len, 2);
    let lr: f32 = 0.0005; // LR más bajo para mayor precisión con ruido

    let start = Instant::now();

    for epoch in 1..=100 {
        let mut total_loss = 0.0;
        for i in 0..x_train.nrows() {
            let input = x_train.index_axis(Axis(0), i).insert_axis(Axis(0)).to_owned();
            let target = y_train.index_axis(Axis(0), i).insert_axis(Axis(0)).to_owned();

            let z1 = conv.forward(&input);
            let a1 = leaky_relu(&z1);
            let p1 = pool.forward(&a1);
            let flat = p1.into_shape((1, num_filters * pool_len)).unwrap();
            let output = dense.forward(&flat);

            let error = &output - &target;
            total_loss += error.mapv(|x| x.powi(2)).sum();

            let grad_dense = dense.backward(&error, lr);
            let grad_p1 = grad_dense.into_shape((num_filters, pool_len)).unwrap();
            let mut grad_a1 = Array2::zeros(a1.dim());
            for f in 0..num_filters {
                for t in 0..pool_len { grad_a1[[f, t * 2]] = grad_p1[[f, t]]; }
            }
            let grad_z1 = &grad_a1 * &leaky_relu_prime(&z1);
            conv.backward(&grad_z1, lr);
        }
        if epoch % 10 == 0 {
            println!("Época {:3} | Loss: {:.6}", epoch, total_loss / 10000.0);
        }
    }

    println!("\n⏱️ Entrenamiento Hardening completado en: {:?}", start.elapsed());
    let model = ModelV6 { conv, dense };
    let f = File::create("modelo_v6.json").unwrap();
    serde_json::to_writer(f, &model).unwrap();
    println!("--- Modelo V6 Guardado ---");
}

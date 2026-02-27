use motor_v3::{RedModular, CapaDensa, Activacion};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::time::Instant;
use std::io::Write;

fn main() {
    let mut red = RedModular::new(0.05);
    red.agregar_capa(CapaDensa::new(4, 32, Activacion::LeakyReLU));
    red.agregar_capa(CapaDensa::new(32, 16, Activacion::LeakyReLU));
    red.agregar_capa(CapaDensa::new(16, 3, Activacion::Sigmoide));

    let n_total = 6000;
    let x_total = Array2::<f64>::random((n_total, 4), Uniform::new(0.0, 1.0));
    let mut y_total = Array2::<f64>::zeros((n_total, 3));
    for i in 0..n_total {
        if x_total[[i,0]] + x_total[[i,1]] > 1.0 { y_total[[i,0]] = 1.0; } else { y_total[[i,1]] = 1.0; }
    }

    let x_train = x_total.slice(s![0..5000, ..]).to_owned();
    let y_train = y_total.slice(s![0..5000, ..]).to_owned();
    let x_test = x_total.slice(s![5000..6000, ..]).to_owned();
    let y_test = y_total.slice(s![5000..6000, ..]).to_owned();

    let _ = std::fs::remove_file("training_log.csv");
    {
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open("training_log.csv").unwrap();
        writeln!(file, "epoca,loss_train,loss_test").unwrap();
    }

    println!("🚀 Entrenando motor modular...");
    let inicio = Instant::now();

    for epoca in 0..1001 {
        let pred_train = red.forward(&x_train);
        let loss_train = (&pred_train - &y_train).mapv(|v| v.powi(2)).mean().unwrap();
        let grad = &pred_train - &y_train;
        red.backward(&grad);

        if epoca % 100 == 0 {
            let pred_test = red.forward(&x_test);
            let loss_test = (&pred_test - &y_test).mapv(|v| v.powi(2)).mean().unwrap();
            
            println!("📅 Época {:>4} | Loss Train: {:.5} | Loss Test: {:.5}", epoca, loss_train, loss_test);
            
            let mut file = std::fs::OpenOptions::new().append(true).open("training_log.csv").unwrap();
            writeln!(file, "{},{},{}", epoca, loss_train, loss_test).unwrap();
        }
    }

    red.guardar("modelo_modular_v3.json").expect("Error al guardar");
    println!("\n✅ Proceso terminado en {:?}", inicio.elapsed());
    println!("💾 Cerebro exportado a 'modelo_modular_v3.json'");
}

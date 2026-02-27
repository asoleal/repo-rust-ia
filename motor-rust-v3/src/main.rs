mod mnist_loader;
use motor_v3::{RedModular, CapaDensa, Activacion};
use ndarray::prelude::*;
use std::time::Instant;

fn main() {
    println!("📦 --- CARGANDO DATASET MNIST ---");
    let x_train = mnist_loader::load_images("data/train-images-idx3-ubyte");
    let y_train = mnist_loader::load_labels("data/train-labels-idx1-ubyte");
    let x_test = mnist_loader::load_images("data/t10k-images-idx3-ubyte");
    let y_test = mnist_loader::load_labels("data/t10k-labels-idx1-ubyte");

    let n_muestras = x_train.nrows();
    let batch_size = 128; // <--- EL CAMBIO CLAVE

    // Subimos un poco el LR para mini-batches
    let mut red = RedModular::new(0.05); 
    red.agregar_capa(CapaDensa::new(784, 128, Activacion::LeakyReLU));
    red.agregar_capa(CapaDensa::new(128, 64, Activacion::LeakyReLU));
    red.agregar_capa(CapaDensa::new(64, 10, Activacion::Sigmoide));

    println!("\n🚀 --- ENTRENAMIENTO CON MINI-BATCHES ---");
    println!("| Época |  Loss  | Accuracy | Tiempo Época |");
    println!("|-------|--------|----------|--------------|");

    let inicio_total = Instant::now();

    for epoca in 0..11 {
        let inicio_epoca = Instant::now();

        // Bucle de Mini-batches
        for i in (0..n_muestras).step_by(batch_size) {
            let fin = (i + batch_size).min(n_muestras);
            let x_batch = x_train.slice(s![i..fin, ..]).to_owned();
            let y_batch = y_train.slice(s![i..fin, ..]).to_owned();

            let pred = red.forward(&x_batch);
            let grad = &pred - &y_batch;
            red.backward(&grad);
        }

        // Evaluación cada época
        let test_pred = red.forward(&x_test);
        let mut correctos = 0;
        for i in 0..test_pred.nrows() {
            let mut max_idx = 0;
            let mut max_val = -1.0;
            for j in 0..10 {
                if test_pred[[i, j]] > max_val {
                    max_val = test_pred[[i, j]];
                    max_idx = j;
                }
            }
            if y_test[[i, max_idx]] == 1.0 { correctos += 1; }
        }
        let acc = (correctos as f64 / x_test.nrows() as f64) * 100.0;
        let loss = (&test_pred - &y_test).mapv(|v| v.powi(2)).mean().unwrap();

        println!("| {:>5} | {:.4} | {:>7.2}% | {:>11.2?} |", epoca, loss, acc, inicio_epoca.elapsed());
    }

    println!("\n✅ Fin: {:.2}s totales.", inicio_total.elapsed().as_secs_f64());
    red.guardar("modelo_mnist_v3.json").unwrap();
}

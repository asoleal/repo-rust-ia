mod mnist_loader;
use motor_v3::{RedModular, CapaDensa, Activacion};
use ndarray::prelude::*;
use rand::seq::SliceRandom;
use std::time::Instant;

fn main() {
    let x_train = mnist_loader::load_images("data/train-images-idx3-ubyte");
    let y_train = mnist_loader::load_labels("data/train-labels-idx1-ubyte");
    let x_test = mnist_loader::load_images("data/t10k-images-idx3-ubyte");
    let y_test = mnist_loader::load_labels("data/t10k-labels-idx1-ubyte");

    let n = x_train.nrows();
    let mut indices: Vec<usize> = (0..n).collect();
    let batch_size = 128;
    
    // Adam requiere un LR mucho más pequeño, típicamente 0.001
    let mut red = RedModular::new(0.001);
    red.agregar_capa(CapaDensa::new(784, 256, Activacion::LeakyReLU));
    red.agregar_capa(CapaDensa::new(256, 10, Activacion::Softmax));

    println!("🔥 Entrenamiento 'Grado 10' iniciado...");

    for epoca in 0..10 {
        indices.shuffle(&mut rand::thread_rng()); // <--- SHUFFLE
        let inicio = Instant::now();

        for chunk in indices.chunks(batch_size) {
            let x_batch = x_train.select(Axis(0), chunk);
            let y_batch = y_train.select(Axis(0), chunk);

            let pred = red.forward(&x_batch);
            let grad = &pred - &y_batch; // Derivada de Softmax + CrossEntropy
            red.backward_adam(&grad);
        }

        // Evaluación
        let test_pred = red.forward(&x_test);
        let mut ok = 0;
        for i in 0..x_test.nrows() {
            let p = test_pred.row(i).iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if y_test[[i, p]] == 1.0 { ok += 1; }
        }
        println!("Época {} | Accuracy: {:.2}% | Tiempo: {:?}", epoca, (ok as f64 / 10000.0)*100.0, inicio.elapsed());
    }
    red.guardar("modelo_mnist_v3.json").unwrap();
}

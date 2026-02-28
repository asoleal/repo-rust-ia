mod mnist_loader;
use motor_rust_v4::{RedCNN, CapaConv2D, CapaPooling, Flatten, CapaDensa};
use ndarray::prelude::*;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("📦 Cargando MNIST...");
    let x_train = mnist_loader::load_images_4d("data/train-images-idx3-ubyte");
    let y_train = mnist_loader::load_labels("data/train-labels-idx1-ubyte");

    let mut red = RedCNN::new();
    let lr = 0.0005;
    red.agregar(Box::new(CapaConv2D::new(16, 1, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(CapaConv2D::new(16, 16, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(Flatten { shape_entrada: None }));
    red.agregar(Box::new(CapaDensa::new(400, 10, lr)));

    println!("🔥 Entrenando 5 épocas...");
    for epoca in 0..5 {
        let mut loss = 0.0;
        for i in (0..60000).step_by(64) {
            let fin = (i + 64).min(60000);
            loss += red.train_step(&x_train.slice(s![i..fin,..,..,..]).to_owned(), &y_train.slice(s![i..fin,..]).to_owned());
        }
        println!("✅ Época {} | Loss: {:.6}", epoca, loss/937.0);
    }

    // GUARDAR PESOS EN BINARIO (Muy básico pero efectivo)
    println!("💾 Guardando pesos...");
    let mut pesos_data = Vec::new();
    for capa in &red.capas {
        if let (Some(w), Some(b)) = capa.get_weights() {
            pesos_data.push((w, b));
        }
    }
    let serialized = serde_json::to_string(&pesos_data).unwrap();
    let mut file = File::create("modelo_v4.json").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
    println!("🏆 ¡Modelo guardado en modelo_v4.json!");
}

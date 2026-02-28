mod mnist_loader;
use motor_rust_v4::{RedCNN, CapaConv2D, CapaPooling, Flatten, CapaDensa};
use ndarray::prelude::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let idx: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);

    let x_test = mnist_loader::load_images_4d("data/t10k-images-idx3-ubyte");
    let y_test = mnist_loader::load_labels("data/t10k-labels-idx1-ubyte");

    let mut red = RedCNN::new();
    let lr = 0.0005;

    // Mismo diseño exacto
    red.agregar(Box::new(CapaConv2D::new(16, 1, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(CapaConv2D::new(16, 16, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(Flatten { shape_entrada: None }));
    red.agregar(Box::new(CapaDensa::new(400, 10, lr)));

    let img = x_test.slice(s![idx..idx+1, .., .., ..]).to_owned();
    let mut salida = img;
    for capa in &mut red.capas { salida = capa.forward(&salida); }
    
    let exp_salida = salida.mapv(|v| v.exp());
    let prob = &exp_salida / exp_salida.sum();
    
    let (mut max_p, mut pred) = (0.0, 0);
    for i in 0..10 {
        if prob[[0, i, 0, 0]] > max_p { max_p = prob[[0, i, 0, 0]]; pred = i; }
    }

    println!("\n------------------------------");
    println!("🔮 Predicción DEEP: {} | Confianza: {:.2}%", pred, max_p * 100.0);
    println!("🎯 Valor Real:      {}", y_test.slice(s![idx, ..]).iter().position(|&x| x == 1.0).unwrap());
    println!("------------------------------");
}

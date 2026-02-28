mod mnist_loader;
use motor_rust_v4::{RedCNN, CapaConv2D, CapaPooling, Flatten, CapaDensa};
use ndarray::prelude::*;
use std::fs::File;
use std::io::Read;

fn main() {
    println!("📦 Cargando datos de prueba (MNIST Test Set)...");
    let x_test = mnist_loader::load_images_4d("data/t10k-images-idx3-ubyte");
    let y_test = mnist_loader::load_labels("data/t10k-labels-idx1-ubyte");
    
    let mut red = RedCNN::new();
    let lr = 0.0005;
    
    // Arquitectura idéntica
    red.agregar(Box::new(CapaConv2D::new(16, 1, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(CapaConv2D::new(16, 16, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(Flatten { shape_entrada: None }));
    red.agregar(Box::new(CapaDensa::new(400, 10, lr)));

    // --- CARGAR INTELIGENCIA ENTRENADA ---
    println!("📂 Cargando pesos desde modelo_v4.json...");
    let mut file = File::open("modelo_v4.json").expect("❌ No se encontró modelo_v4.json. ¡Entrena primero!");
    let mut contenido = String::new();
    file.read_to_string(&mut contenido).unwrap();
    let pesos_cargados: Vec<(Array4<f64>, Array1<f64>)> = serde_json::from_str(&contenido).unwrap();

    let mut idx_p = 0;
    for capa in &mut red.capas {
        if let (Some(_), Some(_)) = capa.get_weights() {
            let (w, b) = &pesos_cargados[idx_p];
            capa.set_weights(w.clone(), b.clone());
            idx_p += 1;
        }
    }

    let mut aciertos = 0;
    let total = 2000; // Evaluaremos 2000 imágenes para una estadística sólida
    println!("🏟️  Iniciando Arena: Evaluando {} imágenes desconocidas...", total);

    for i in 0..total {
        let img = x_test.slice(s![i..i+1, .., .., ..]).to_owned();
        let mut salida = img;
        for capa in &mut red.capas { salida = capa.forward(&salida); }
        
        let pred = salida.slice(s![0, .., 0, 0]).iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(idx, _)| idx).unwrap();
            
        let real = y_test.slice(s![i, ..]).iter().position(|&x| x == 1.0).unwrap();
        
        if pred == real { aciertos += 1; }
        if i % 200 == 0 { print!("•"); std::io::Write::flush(&mut std::io::stdout()).unwrap(); }
    }

    println!("\n---------------------------------------");
    println!("📊 PRECISIÓN FINAL (Accuracy): {:.2}%", (aciertos as f64 / total as f64) * 100.0);
    println!("---------------------------------------");
}

mod mnist_loader;
use motor_rust_v4::{RedCNN, CapaConv2D, CapaPooling, Flatten, CapaDensa};
use ndarray::prelude::*;
use image::io::Reader as ImageReader;
use std::fs::File;
use std::io::Read;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let ruta = args.get(1).expect("Uso: cargo run --release --bin ver_imagen -- imagen.png");

    let mut red = RedCNN::new();
    let lr = 0.0005;
    red.agregar(Box::new(CapaConv2D::new(16, 1, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(CapaConv2D::new(16, 16, 3, lr)));
    red.agregar(Box::new(CapaPooling { size: 2 }));
    red.agregar(Box::new(Flatten { shape_entrada: None }));
    red.agregar(Box::new(CapaDensa::new(400, 10, lr)));

    // CARGAR PESOS
    println!("📂 Cargando inteligencia desde modelo_v4.json...");
    let mut file = File::open("modelo_v4.json").expect("No se encontró el modelo. ¡Entrena primero!");
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

    // PROCESAR IMAGEN
    let img = ImageReader::open(ruta).unwrap().decode().unwrap()
        .resize_exact(28, 28, image::imageops::FilterType::Lanczos3).to_luma8();
    let mut input = Array4::<f64>::zeros((1, 1, 28, 28));
    for (x, y, p) in img.enumerate_pixels() {
        input[[0, 0, y as usize, x as usize]] = p[0] as f64 / 255.0;
    }

    let mut salida = input;
    for capa in &mut red.capas { salida = capa.forward(&salida); }
    let (mut max_p, mut pred) = (0.0, 0);
    let prob = &salida / salida.sum(); // Softmax simple
    for i in 0..10 {
        if prob[[0, i, 0, 0]] > max_p { max_p = prob[[0, i, 0, 0]]; pred = i; }
    }

    println!("\n🔮 Predicción: {} | Confianza: {:.2}%", pred, max_p * 100.0);
}

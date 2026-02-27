use motor_v3::{RedModular, mnist_loader};
use ndarray::prelude::*;
use std::env;

fn dibujar_ascii(imagen: &ArrayView1<f64>) {
    for r in 0..28 {
        for c in 0..28 {
            let pixel = imagen[r * 28 + c];
            let char = match pixel {
                p if p > 0.8 => "█",
                p if p > 0.4 => "▒",
                p if p > 0.1 => "░",
                _ => " ",
            };
            print!("{}", char);
        }
        println!();
    }
}

fn main() {
    // El cargador ahora inicializa la estructura y mete las capas
    let red = RedModular::cargar("modelo_mnist_v3.json");

    let x_test = mnist_loader::load_images("data/t10k-images-idx3-ubyte");
    let y_test = mnist_loader::load_labels("data/t10k-labels-idx1-ubyte");

    let idx = env::args().nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(500);

    let imagen = x_test.slice(s![idx, ..]);
    let etiqueta_real = y_test.slice(s![idx, ..]);

    println!("\n🎨 Visualización de entrada (Indice {}):", idx);
    dibujar_ascii(&imagen);

    // En el forward ahora usamos la red cargada
    let mut red_clon = red; 
    let prediccion = red_clon.forward(&imagen.to_owned().insert_axis(Axis(0)));

    let (pred_digit, &confianza) = prediccion.row(0).iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();

    let real_digit = etiqueta_real.iter().position(|&x| x == 1.0).unwrap();

    println!("\n------------------------------");
    println!("🔮 Predicción (Softmax): {} ", pred_digit);
    println!("🎯 Valor Real:          {} ", real_digit);
    println!("📈 Probabilidad:        {:.4}%", confianza * 100.0);
    println!("------------------------------\n");
}

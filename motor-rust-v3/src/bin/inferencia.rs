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
    let mut red = RedModular::cargar("modelo_mnist_v3.json")
        .expect("❌ Error al cargar modelo");

    let x_test = mnist_loader::load_images("data/t10k-images-idx3-ubyte");
    let y_test = mnist_loader::load_labels("data/t10k-labels-idx1-ubyte");

    let idx = env::args().nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(500);

    let imagen = x_test.slice(s![idx, ..]);
    let etiqueta_real = y_test.slice(s![idx, ..]);

    println!("\n🎨 Visualización de la entrada:");
    dibujar_ascii(&imagen);

    let prediccion = red.forward(&imagen.to_owned().insert_axis(Axis(0)));

    let mut max_val = -1.0;
    let mut pred_digit = 0;
    for j in 0..10 {
        if prediccion[[0, j]] > max_val {
            max_val = prediccion[[0, j]];
            pred_digit = j;
        }
    }

    let real_digit = etiqueta_real.iter().position(|&x| x == 1.0).unwrap();

    println!("\n------------------------------");
    println!("🖼️  Imagen Índice: {}", idx);
    println!("🔮 Predicción:      {}", pred_digit);
    println!("🎯 Valor Real:      {}", real_digit);
    println!("📈 Confianza:       {:.2}%", max_val * 100.0);
    println!("------------------------------\n");
}

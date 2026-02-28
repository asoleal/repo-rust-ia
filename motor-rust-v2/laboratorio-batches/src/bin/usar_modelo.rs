use laboratorio_batches::RedNativa;
use ndarray::prelude::*;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📂 Cargando cerebro desde archivo...");
    let red = RedNativa::cargar("modelo_pesado.json")?;
    println!("🧠 Modelo cargado. Listo para predecir.");

    loop {
        print!("> "); io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let values: Vec<f64> = input.split_whitespace().map(|s| s.parse().unwrap_or(0.0)).collect();
        if values.len() != 4 { break; }

        let x = Array2::from_shape_vec((1, 4), values)?;
        let mut a = x.clone();
        for (w, b) in red.pesos.iter().zip(red.sesgos.iter()) {
            let z = a.dot(w) + b;
            a = z.mapv(|val| 1.0 / (1.0 + (-val).exp()));
        }
        println!("✅ Predicción: {:?}\n", a.row(0));
    }
    Ok(())
}

use motor_rust_native::RedNativa;
use ndarray::prelude::*;
use std::error::Error;
use std::time::Instant;

fn cargar_csv(path: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for result in reader.records() {
        let record = result?;
        let feat: Vec<f64> = record
            .iter()
            .take(4)
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();
        inputs.push(feat);

        let label: usize = record.get(4).unwrap().parse()?;
        let mut one_hot = vec![0.0; 3];
        one_hot[label] = 1.0;
        targets.push(one_hot);
    }

    let rows = inputs.len();
    let x = Array2::from_shape_vec((rows, 4), inputs.into_iter().flatten().collect())?;
    let y = Array2::from_shape_vec((rows, 3), targets.into_iter().flatten().collect())?;
    Ok((x, y))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("📖 Cargando dataset Iris desde CSV...");
    let (x, y) = cargar_csv("iris.csv")?;

    // Arquitectura: 4 entradas, 8 neuronas ocultas, 3 salidas
    let mut red = RedNativa::new_rust(vec![4, 8, 3], 0.05);

    println!("🚀 Entrenando...");
    let inicio = Instant::now();
    for epoca in 0..5000 {
        let _loss = red.train_native(&x, &y);
    }
    println!("⏱️ Entrenamiento completado en {:?}", inicio.elapsed());

    // --- AQUÍ DEBEN IR LAS LÍNEAS DE GUARDADO ---
    red.save("iris_entrenado_nativo.json")?;
    println!("💾 Modelo guardado con éxito como 'iris_entrenado_nativo.json'");
    // --------------------------------------------

    Ok(())
}

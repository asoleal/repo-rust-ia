use motor_v3::{RedModular, CapaDensa, Activacion};
use ndarray::prelude::*;
use std::sync::Arc;
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let mut red_base = RedModular::new(0.05);
    red_base.agregar_capa(CapaDensa::new(4, 128, Activacion::LeakyReLU));
    red_base.agregar_capa(CapaDensa::new(128, 64, Activacion::LeakyReLU));
    red_base.agregar_capa(CapaDensa::new(64, 3, Activacion::Sigmoide));

    // Envolvemos la red en un Arc para compartirla entre hilos
    let red_compartida = Arc::new(red_base);

    println!("🔥 --- BENCHMARK HOGWILD (8 Hilos) ---");

    let batch_size = 20000;
    let x = Array2::<f64>::zeros((batch_size, 4));
    let y = Array2::<f64>::zeros((batch_size, 3));

    let inicio = Instant::now();
    let iteraciones = 100;

    for _ in 0..iteraciones {
        // Dividimos el trabajo
        let chunks_x: Vec<_> = x.axis_chunks_iter(Axis(0), batch_size / 8).collect();
        let chunks_y: Vec<_> = y.axis_chunks_iter(Axis(0), batch_size / 8).collect();

        chunks_x.into_par_iter().zip(chunks_y.into_par_iter()).for_each(|(cx, cy)| {
            let r = Arc::clone(&red_compartida);
            // Cada hilo hace forward y calcula gradientes
            let (pred, entradas, zs) = r.forward_thread_safe(&cx.to_owned());
            let grad = &pred - &cy.to_owned();
            
            // Actualización asíncrona Hogwild!
            unsafe { r.update_hogwild(&grad, &entradas, &zs); }
        });
    }

    let duracion = inicio.elapsed().as_secs_f64();
    let total_muestras = (batch_size * iteraciones) as f64;
    println!("🏆 Throughput HOGWILD: {:.2} muestras/seg", total_muestras / duracion);
    println!("⏱️ Tiempo para {} millones: {:.4} s", total_muestras / 1_000_000.0, duracion);
}

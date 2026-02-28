use laboratorio_batches::RedNativa;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::error::Error;
use std::time::Instant;
use rayon::prelude::*;

struct HogwildWrapper { pub red: RedNativa }
unsafe impl Sync for HogwildWrapper {}

fn main() -> Result<(), Box<dyn Error>> {
    let n_muestras = 1_000_000;
    let dist = Uniform::new(0.0, 1.0);
    let x_total = Array2::<f64>::random((n_muestras, 4), dist);
    let y_total = Array2::<f64>::random((n_muestras, 3), dist);
    let wrapper = HogwildWrapper { red: RedNativa::new_rust(vec![4, 32, 3], 0.01) };
    
    println!("🚀 Entrenando y guardando modelo...");
    (0..5).for_each(|_| {
        (0..n_muestras).into_par_iter().step_by(1024).for_each(|i| {
            let fin = std::cmp::min(i + 1024, n_muestras);
            unsafe {
                let red_ptr = &wrapper.red as *const RedNativa as *mut RedNativa;
                (*red_ptr).train_hogwild(x_total.slice(s![i..fin, ..]), y_total.slice(s![i..fin, ..]));
            }
        });
    });

    wrapper.red.guardar("modelo_pesado.json")?;
    println!("💾 ¡Modelo guardado con éxito como 'modelo_pesado.json'!");
    Ok(())
}

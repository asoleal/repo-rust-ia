use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

pub fn generate_sensor_data(samples: usize, seq_len: usize) -> (Array2<f64>, Array2<f64>) {
    let mut rng = thread_rng();
    let mut x = Array2::zeros((samples, seq_len));
    let mut y = Array2::zeros((samples, 2));
    
    // Ruido base de fondo (simula interferencia eléctrica)
    let normal_dist = Normal::new(0.0, 0.20).unwrap(); 

    for i in 0..samples {
        let is_anomaly = rng.gen_bool(0.5);
        let mut signal = Array1::zeros(seq_len);
        
        // Frecuencias: Normal (0.05), Falla (0.18)
        let freq = if is_anomaly { 0.18 } else { 0.05 }; 
        let amplitude = if is_anomaly { 1.2 } else { 1.0 };

        // AGREGADO V6: Señal fantasma de interferencia (0.4 Hz)
        let ghost_freq = 0.40;
        let ghost_amp = rng.gen_range(0.05..0.15);

        for t in 0..seq_len {
            let base = amplitude * (2.0 * PI * freq * t as f64).sin();
            let ghost = ghost_amp * (2.0 * PI * ghost_freq * t as f64).sin();
            let noise = normal_dist.sample(&mut rng);
            
            signal[t] = base + ghost + noise;
        }
        
        x.row_mut(i).assign(&signal);
        if is_anomaly { y[[i, 1]] = 1.0; } else { y[[i, 0]] = 1.0; }
    }
    (x, y)
}

use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use rand::prelude::*;

pub fn generate_sensor_data(samples: usize, seq_len: usize) -> (Array2<f64>, Array2<f64>) {
    let mut rng = thread_rng();
    let mut x = Array2::zeros((samples, seq_len));
    let mut y = Array2::zeros((samples, 2));

    for i in 0..samples {
        let is_anomaly = rng.gen_bool(0.5);
        let mut signal = Array1::zeros(seq_len);
        
        let freq = if is_anomaly { 0.15 } else { 0.05 }; // Diferença de frequência clara
        let amplitude = if is_anomaly { 1.5 } else { 1.0 };

        for t in 0..seq_len {
            let base = amplitude * (2.0 * PI * freq * t as f64).sin();
            let noise = rng.gen_range(-0.2..0.2);
            signal[t] = base + noise;
        }
        
        x.row_mut(i).assign(&signal);
        if is_anomaly { y[[i, 1]] = 1.0; } else { y[[i, 0]] = 1.0; }
    }
    (x, y)
}

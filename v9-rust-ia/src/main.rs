mod nn;
mod model;

use ndarray::{Array2, Axis};
use nn::conv1d::Conv1D;
use nn::dense::Dense;
use nn::ops::{leaky_relu, leaky_relu_prime, AvgPool1D};

use model::{save_f32, Model};

fn softmax3(z: [f32; 3]) -> [f32; 3] {
    let m = z[0].max(z[1]).max(z[2]);
    let e0 = (z[0] - m).exp();
    let e1 = (z[1] - m).exp();
    let e2 = (z[2] - m).exp();
    let s = e0 + e1 + e2;
    [e0 / s, e1 / s, e2 / s]
}

// Genera secuencias que imitan distancia: base + slope*t + ruido + outliers
// Clases: 0 quieto, 1 acercando (slope < 0), 2 alejando (slope > 0)
fn make_ultra_synth(n: usize, seq_len: usize) -> (Array2<f32>, Array2<f32>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut x = Array2::<f32>::zeros((n, seq_len));
    let mut y = Array2::<f32>::zeros((n, 3));

    for i in 0..n {
        let class = i % 3;

        let base_cm: f32 = rng.gen_range(10.0..150.0);

        let slope_cm_per_step: f32 = match class {
            0 => rng.gen_range(-0.02..0.02),      // quieto ~0
            1 => rng.gen_range(-0.35..-0.08),     // acercando
            _ => rng.gen_range(0.08..0.35),       // alejando
        };

        for t in 0..seq_len {
            let mut cm = base_cm + slope_cm_per_step * (t as f32);

            // Ruido tipo ultrasonido
            cm += rng.gen_range(-1.0..1.0);

            // Outlier ocasional
            if rng.gen_bool(0.03) {
                cm += rng.gen_range(-25.0..25.0);
            }

            // Clamp físico razonable
            cm = cm.clamp(2.0, 200.0);

            // Normaliza 0..200cm -> 0..1 (igual que en el ESP32)
            x[[i, t]] = cm / 200.0;
        }

        y[[i, class]] = 1.0;
    }

    (x, y)
}

fn main() {
    let seq_len = 128;
    let num_filters = 12;
    let kernel_size = 5;

    let pool = AvgPool1D::new(2);
    let conv_out_len = (seq_len - kernel_size) + 1; // stride=1
    let pool_len = conv_out_len / 2;                // 62
    let dense_in = num_filters * pool_len;          // 744

    let (x_train, y_train) = make_ultra_synth(3000, seq_len);

    let conv = Conv1D::new(num_filters, 1, kernel_size);
    let dense = Dense::new(dense_in, 3);
    let mut model = Model { conv, dense };

    let lr: f32 = 0.001;

    for epoch in 1..=25 {
        let mut total_loss = 0.0;

        for i in 0..x_train.nrows() {
            let input = x_train
                .index_axis(Axis(0), i)
                .insert_axis(Axis(0))
                .to_owned(); // (1,128)
            let target = y_train
                .index_axis(Axis(0), i)
                .insert_axis(Axis(0))
                .to_owned(); // (1,3)

            let z1 = model.conv.forward(&input);
            let a1 = leaky_relu(&z1);
            let p1 = pool.forward(&a1);
            let flat = p1.into_shape((1, dense_in)).unwrap();

            let logits = model.dense.forward(&flat); // (1,3)

            // Softmax + cross-entropy (solo para loss/grad simple)
            let z = [logits[[0, 0]], logits[[0, 1]], logits[[0, 2]]];
            let p = softmax3(z);

            // Loss CE: -sum(y*log(p))
            let mut ce = 0.0;
            for k in 0..3 {
                let yk = target[[0, k]];
                if yk > 0.0 {
                    ce -= (p[k].max(1e-9)).ln();
                }
            }
            total_loss += ce;

            // Gradiente de CE con softmax: grad = p - y (en logits)
            let mut grad_out = Array2::<f32>::zeros((1, 3));
            for k in 0..3 {
                grad_out[[0, k]] = p[k] - target[[0, k]];
            }

            let grad_flat = model.dense.backward(&grad_out, lr); // (1,744)
            let grad_p1 = grad_flat.into_shape((num_filters, pool_len)).unwrap();

            let grad_a1 = pool.backward(&grad_p1, a1.dim().1);
            let grad_z1 = &grad_a1 * &leaky_relu_prime(&z1);

            model.conv.backward(&grad_z1, lr);
        }

        if epoch % 5 == 0 {
            println!(
                "epoch {:02} loss_ce {:.6}",
                epoch,
                total_loss / (x_train.nrows() as f32)
            );
        }
    }

    // Guardar model.bin (ojo: ahora dense_out=3, así que el tamaño cambia)
    let flat = model.to_flat_f32();
    save_f32("model.bin", &flat).unwrap();
    println!("saved model.bin ({} f32 = {} bytes)", flat.len(), flat.len() * 4);

    // Para tu arquitectura: conv 60 + convb 12 + dense 744*3=2232 + denseb 3 => 2307 f32
    println!("expected f32 = {}", 60 + 12 + (744 * 3) + 3);
}

use anyhow::Result;
use esp_idf_svc::hal::{
    delay::{Ets, FreeRtos},
    gpio::PinDriver,
    peripherals::Peripherals,
};
use esp_idf_svc::nvs::{EspDefaultNvsPartition, EspNvs, NvsDefault};
use esp_idf_svc::sys::{esp_timer_get_time, link_patches};

use std::io::{self, Read};

fn micros() -> i64 {
    unsafe { esp_timer_get_time() }
}

fn bytes_to_f32_le(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0);
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn f32_to_bytes_le(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { x * 0.01 }
}

fn conv1d_forward(
    weights: &[f32],
    bias: &[f32],
    filters: usize,
    in_ch: usize,
    k: usize,
    stride: usize,
    input: &[f32],
    in_len: usize,
    out: &mut [f32],
) {
    let out_len = (in_len - k) / stride + 1;
    for f in 0..filters {
        for t in 0..out_len {
            let mut sum = bias[f];
            for c in 0..in_ch {
                for kk in 0..k {
                    let x = input[c * in_len + (t * stride + kk)];
                    let w = weights[(f * in_ch + c) * k + kk];
                    sum += x * w;
                }
            }
            out[f * out_len + t] = sum;
        }
    }
}

fn avgpool1d_forward(
    size: usize,
    input: &[f32],
    filters: usize,
    in_len: usize,
    out: &mut [f32],
) {
    let out_len = in_len / size;
    let scale = 1.0 / (size as f32);
    for f in 0..filters {
        for i in 0..out_len {
            let start = i * size;
            let mut sum = 0.0;
            for j in 0..size {
                sum += input[f * in_len + (start + j)];
            }
            out[f * out_len + i] = sum * scale;
        }
    }
}

fn dense_forward(
    weights: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
    input: &[f32],
    out: &mut [f32],
) {
    for j in 0..out_dim {
        let mut sum = bias[j];
        for i in 0..in_dim {
            sum += input[i] * weights[i * out_dim + j];
        }
        out[j] = sum;
    }
}

fn softmax3(z: [f32; 3]) -> [f32; 3] {
    let m = z[0].max(z[1]).max(z[2]);
    let e0 = (z[0] - m).exp();
    let e1 = (z[1] - m).exp();
    let e2 = (z[2] - m).exp();
    let s = e0 + e1 + e2;
    [e0 / s, e1 / s, e2 / s]
}

fn argmax3(a: f32, b: f32, c: f32) -> usize {
    if a >= b && a >= c { 0 } else if b >= a && b >= c { 1 } else { 2 }
}

fn class_name(c: usize) -> &'static str {
    match c {
        0 => "quieto",
        1 => "acercando",
        2 => "alejando",
        _ => "?",
    }
}

fn read_ultrasonic_cm(
    trig: &mut PinDriver<'static, esp_idf_svc::hal::gpio::Gpio23, esp_idf_svc::hal::gpio::Output>,
    echo: &PinDriver<'static, esp_idf_svc::hal::gpio::Gpio2, esp_idf_svc::hal::gpio::Input>,
) -> Result<Option<f32>> {
    trig.set_low()?;
    Ets::delay_us(2);
    trig.set_high()?;
    Ets::delay_us(10);
    trig.set_low()?;

    let t0 = micros();
    while echo.is_low() && (micros() - t0) < 30_000 {}

    let start = micros();
    while echo.is_high() && (micros() - start) < 30_000 {}
    let end = micros();

    let pulse_us = (end - start).max(0) as f32;
    if pulse_us <= 1.0 {
        return Ok(None);
    }
    Ok(Some(pulse_us / 58.0))
}

// Devuelve:
// - Some(Ok(label_0_2)) si el usuario tecleó 1/2/3
// - Some(Err(())) si tecleó 't' para toggle train
// - None si no llegó nada útil
fn poll_cmd_from_stdin() -> Option<Result<usize, ()>> {
    let mut buf = [0u8; 64];
    if let Ok(n) = io::stdin().read(&mut buf) {
        if n == 0 {
            return None;
        }
        let s = String::from_utf8_lossy(&buf[..n]);
        for ch in s.chars() {
            match ch {
                't' | 'T' => return Some(Err(())),
                '1' => return Some(Ok(0)), // 1 -> clase 0
                '2' => return Some(Ok(1)), // 2 -> clase 1
                '3' => return Some(Ok(2)), // 3 -> clase 2
                _ => {}
            }
        }
    }
    None
}

fn main() -> Result<()> {
    link_patches();

    static MODEL_BIN: &[u8] = include_bytes!("../../assets/model.bin");
    let flat = bytes_to_f32_le(MODEL_BIN);

    let seq_len = 128usize;
    let filters = 12usize;
    let in_ch = 1usize;
    let k = 5usize;
    let stride = 1usize;
    let pool_size = 2usize;

    let conv_out_len = (seq_len - k) / stride + 1; // 124
    let pool_len = conv_out_len / pool_size;       // 62
    let dense_in = filters * pool_len;             // 744
    let dense_out = 3usize;

    let conv_w_len = filters * in_ch * k;          // 60
    let conv_b_len = filters;                      // 12
    let dense_w_len = dense_in * dense_out;        // 2232
    let dense_b_len = dense_out;                   // 3
    let total_need = conv_w_len + conv_b_len + dense_w_len + dense_b_len;

    println!("ia_train: model bytes={} flat={} need={}", MODEL_BIN.len(), flat.len(), total_need);
    if flat.len() != total_need {
        println!("ia_train: ERROR tamaño de modelo inesperado");
        loop { FreeRtos::delay_ms(1000); }
    }

    let mut idx = 0;
    let conv_w = &flat[idx..idx + conv_w_len]; idx += conv_w_len;
    let conv_b = &flat[idx..idx + conv_b_len]; idx += conv_b_len;

    let mut dense_w: Vec<f32> = flat[idx..idx + dense_w_len].to_vec(); idx += dense_w_len;
    let mut dense_b: Vec<f32> = flat[idx..idx + dense_b_len].to_vec();

    let nvs_part = EspDefaultNvsPartition::take()?;
    let mut nvs: EspNvs<NvsDefault> = EspNvs::new(nvs_part, "ia", true)?;

    const KEY_DENSE: &str = "dense_v1";
    let expected_bytes = (dense_w_len + dense_b_len) * 4;

    match nvs.blob_len(KEY_DENSE)? {
        Some(len) if len == expected_bytes => {
            let mut bytes = vec![0u8; len];
            if let Some(got) = nvs.get_blob(KEY_DENSE, &mut bytes)? {
                let vals = bytes_to_f32_le(got);
                if vals.len() == dense_w_len + dense_b_len {
                    dense_w.copy_from_slice(&vals[..dense_w_len]);
                    dense_b.copy_from_slice(&vals[dense_w_len..]);
                    println!("ia_train: loaded dense from NVS ({} bytes)", len);
                } else {
                    println!("ia_train: NVS dense decode size mismatch (vals={})", vals.len());
                }
            } else {
                println!("ia_train: NVS key exists but read returned None");
            }
        }
        Some(len) => {
            println!("ia_train: NVS dense exists but size mismatch ({} bytes)", len);
        }
        None => {
            println!("ia_train: no dense in NVS yet");
        }
    }

    let peripherals = Peripherals::take().unwrap();
    let mut trig = PinDriver::output(peripherals.pins.gpio23)?;
    let echo = PinDriver::input(peripherals.pins.gpio2)?;

    let mut x = [0f32; 128];
    let mut z1 = vec![0f32; filters * conv_out_len];
    let mut a1 = vec![0f32; filters * conv_out_len];
    let mut p1 = vec![0f32; filters * pool_len];
    let mut flat_in = vec![0f32; dense_in];
    let mut out = vec![0f32; dense_out];

    let lr: f32 = 0.01;

    let mut training_mode = false;
    println!("ia_train:");
    println!("  - Presiona 't' para START/STOP entrenamiento");
    println!("  - En entrenamiento, presiona 1/2/3 para etiquetar la última ventana y actualizar Dense");
    println!("  - Fuera de entrenamiento, solo hace inferencia");

    loop {
        let mut last_cm = 0.0f32;
        let mut valid = 0u32;

        for i in 0..seq_len {
            match read_ultrasonic_cm(&mut trig, &echo)? {
                Some(cm) => {
                    last_cm = cm;
                    valid += 1;
                    let cm_clamped = cm.clamp(2.0, 200.0);
                    x[i] = cm_clamped / 200.0;
                }
                None => {
                    x[i] = x[i.saturating_sub(1)];
                }
            }
            FreeRtos::delay_ms(40);
        }

        conv1d_forward(conv_w, conv_b, filters, in_ch, k, stride, &x, seq_len, &mut z1);
        for i in 0..a1.len() { a1[i] = leaky_relu(z1[i]); }
        avgpool1d_forward(pool_size, &a1, filters, conv_out_len, &mut p1);
        flat_in.copy_from_slice(&p1);

        dense_forward(&dense_w, &dense_b, dense_in, dense_out, &flat_in, &mut out);

        let cls = argmax3(out[0], out[1], out[2]);
        println!(
            "dist_cm={:.1} valid={}/{} out=[{:.3},{:.3},{:.3}] class={} ({}) train={}",
            last_cm, valid, seq_len, out[0], out[1], out[2], cls, class_name(cls), training_mode
        );

        if let Some(cmd) = poll_cmd_from_stdin() {
            match cmd {
                Err(()) => {
                    training_mode = !training_mode;
                    println!("ia_train: training_mode={}", training_mode);
                }
                Ok(label) => {
                    if !training_mode {
                        println!("ia_train: ignorado (no estás en modo entrenamiento). Presiona 't' para activar.");
                    } else {
                        let p = softmax3([out[0], out[1], out[2]]);
                        let mut y = [0f32; 3];
                        y[label] = 1.0;

                        let g0 = p[0] - y[0];
                        let g1 = p[1] - y[1];
                        let g2 = p[2] - y[2];

                        for i in 0..dense_in {
                            let xi = flat_in[i];
                            let base = i * dense_out;
                            dense_w[base + 0] -= lr * xi * g0;
                            dense_w[base + 1] -= lr * xi * g1;
                            dense_w[base + 2] -= lr * xi * g2;
                        }
                        dense_b[0] -= lr * g0;
                        dense_b[1] -= lr * g1;
                        dense_b[2] -= lr * g2;

                        let mut vals = Vec::with_capacity(dense_w_len + dense_b_len);
                        vals.extend_from_slice(&dense_w);
                        vals.extend_from_slice(&dense_b);
                        let bytes = f32_to_bytes_le(&vals);
                        nvs.set_blob(KEY_DENSE, &bytes)?;
                        println!("ia_train: trained+saved ({} bytes) label={}", bytes.len(), label + 1);
                    }
                }
            }
        }
    }
}

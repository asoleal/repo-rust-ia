use esp_idf_svc::hal::{
    delay::{Ets, FreeRtos},
    gpio::PinDriver,
    peripherals::Peripherals,
};
use esp_idf_svc::sys::{esp_timer_get_time, link_patches};

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
) -> anyhow::Result<Option<f32>> {
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

fn main() -> anyhow::Result<()> {
    link_patches();

    static MODEL_BIN: &[u8] = include_bytes!("../../assets/model.bin");
    println!("ia_test 3c: model bytes={}", MODEL_BIN.len());

    let flat = bytes_to_f32_le(MODEL_BIN);

    // Arquitectura fija (de tu training en PC)
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

    // Layout del flat:
    // conv_w(60) + conv_b(12) + dense_w(744*3=2232) + dense_b(3) = 2307
    let conv_w_len = filters * in_ch * k;
    let conv_b_len = filters;
    let dense_w_len = dense_in * dense_out;
    let dense_b_len = dense_out;
    let need = conv_w_len + conv_b_len + dense_w_len + dense_b_len;

    println!("ia_test 3c: flat.len={} need={}", flat.len(), need);
    if flat.len() != need {
        println!("ia_test 3c: ERROR tamaño de modelo inesperado");
        loop { FreeRtos::delay_ms(1000); }
    }

    let mut idx = 0;
    let conv_w = &flat[idx..idx + conv_w_len]; idx += conv_w_len;
    let conv_b = &flat[idx..idx + conv_b_len]; idx += conv_b_len;
    let dense_w = &flat[idx..idx + dense_w_len]; idx += dense_w_len;
    let dense_b = &flat[idx..idx + dense_b_len];

    let peripherals = Peripherals::take().unwrap();
    let mut trig = PinDriver::output(peripherals.pins.gpio23)?;
    let echo = PinDriver::input(peripherals.pins.gpio2)?;

    let mut x = [0f32; 128];
    let mut z1 = vec![0f32; filters * conv_out_len];
    let mut a1 = vec![0f32; filters * conv_out_len];
    let mut p1 = vec![0f32; filters * pool_len];
    let mut flat_in = vec![0f32; dense_in];
    let mut out = vec![0f32; dense_out];

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
            FreeRtos::delay_ms(60);
        }

        conv1d_forward(conv_w, conv_b, filters, in_ch, k, stride, &x, seq_len, &mut z1);
        for i in 0..a1.len() { a1[i] = leaky_relu(z1[i]); }
        avgpool1d_forward(pool_size, &a1, filters, conv_out_len, &mut p1);
        flat_in.copy_from_slice(&p1);
        dense_forward(dense_w, dense_b, dense_in, dense_out, &flat_in, &mut out);

        let cls = argmax3(out[0], out[1], out[2]);

        println!(
            "dist_cm={:.1} valid={}/{} out=[{:.3},{:.3},{:.3}] class={} ({})",
            last_cm, valid, seq_len, out[0], out[1], out[2], cls, class_name(cls)
        );
    }
}

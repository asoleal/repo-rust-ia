use esp_idf_svc::sys::link_patches;

fn bytes_to_f32_le(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0);
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn main() {
    link_patches();

    static MODEL_BIN: &[u8] = include_bytes!("../assets/model.bin");

    println!("model.bin bytes={}", MODEL_BIN.len());

    let flat = bytes_to_f32_le(MODEL_BIN);
    println!("model.bin f32_count={}", flat.len());

    if flat.len() == 1562 {
        println!("OK: tamaño esperado (1562 f32)");
    } else {
        println!("WARN: tamaño inesperado, esperado 1562 f32");
    }

    loop {
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}

use std::fs::File;
use std::io::Read;
use ndarray::prelude::*;

pub fn load_images(path: &str) -> Array2<f64> {
    let mut f = File::open(path).expect("❌ Error al abrir imágenes");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let n_images = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let mut data = Vec::with_capacity(n_images * 784);
    for i in 16..buf.len() {
        data.push(buf[i] as f64 / 255.0);
    }
    Array2::from_shape_vec((n_images, 784), data).unwrap()
}

pub fn load_labels(path: &str) -> Array2<f64> {
    let mut f = File::open(path).expect("❌ Error al abrir etiquetas");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let n_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let mut labels = Array2::zeros((n_labels, 10));
    for i in 0..n_labels {
        let label = buf[i + 8] as usize;
        if label < 10 { labels[[i, label]] = 1.0; }
    }
    labels
}

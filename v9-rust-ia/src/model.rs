use crate::nn::{conv1d::Conv1D, dense::Dense};
use ndarray::{Array1, Array2, Array3};

pub struct Model {
    pub conv: Conv1D,
    pub dense: Dense,
}

impl Model {
    pub fn to_flat_f32(&self) -> Vec<f32> {
        let mut v = Vec::new();

        v.extend(self.conv.weights.iter().copied());
        v.extend(self.conv.bias.iter().copied());

        v.extend(self.dense.weights.iter().copied());
        v.extend(self.dense.bias.iter().copied());

        v
    }

    pub fn from_flat_f32(
        flat: &[f32],
        conv_shape: (usize, usize, usize), // (filters, in_ch, k)
        dense_shape: (usize, usize),       // (in_dim, out_dim)
    ) -> Self {
        let (filt, in_ch, k) = conv_shape;
        let (din, dout) = dense_shape;

        let conv_w_len = filt * in_ch * k;
        let conv_b_len = filt;
        let dense_w_len = din * dout;
        let dense_b_len = dout;

        let need = conv_w_len + conv_b_len + dense_w_len + dense_b_len;
        assert!(
            flat.len() >= need,
            "flat too small: have {}, need {}",
            flat.len(),
            need
        );

        let mut i = 0;

        let conv_w = Array3::from_shape_vec(
            (filt, in_ch, k),
            flat[i..i + conv_w_len].to_vec(),
        )
        .unwrap();
        i += conv_w_len;

        let conv_b = Array1::from_vec(flat[i..i + conv_b_len].to_vec());
        i += conv_b_len;

        let dense_w = Array2::from_shape_vec(
            (din, dout),
            flat[i..i + dense_w_len].to_vec(),
        )
        .unwrap();
        i += dense_w_len;

        let dense_b = Array1::from_vec(flat[i..i + dense_b_len].to_vec());

        let mut conv = Conv1D::new(filt, in_ch, k);
        conv.weights = conv_w;
        conv.bias = conv_b;

        let mut dense = Dense::new(din, dout);
        dense.weights = dense_w;
        dense.bias = dense_b;

        Self { conv, dense }
    }
}

pub fn save_f32(path: &str, data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for &x in data {
        f.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_f32(path: &str) -> std::io::Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    if bytes.len() % 4 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "len % 4 != 0",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

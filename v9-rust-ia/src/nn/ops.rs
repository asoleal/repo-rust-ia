use ndarray::Array2;

pub fn leaky_relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|val| if val > 0.0 { val } else { val * 0.01 })
}

pub fn leaky_relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|val| if val > 0.0 { 1.0 } else { 0.01 })
}

pub struct AvgPool1D {
    pub size: usize,
}

impl AvgPool1D {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    // input: (filters, len) -> output: (filters, len/size)
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (filters, len) = input.dim();
        let out_len = len / self.size;
        let mut output = Array2::zeros((filters, out_len));

        for f in 0..filters {
            for i in 0..out_len {
                let start = i * self.size;
                let mut sum = 0.0;
                for j in 0..self.size {
                    sum += input[[f, start + j]];
                }
                output[[f, i]] = sum / (self.size as f32);
            }
        }
        output
    }

    // grad_output: (filters, out_len) -> grad_input: (filters, input_len)
    pub fn backward(&self, grad_output: &Array2<f32>, input_len: usize) -> Array2<f32> {
        let (filters, out_len) = grad_output.dim();
        let mut grad_input = Array2::zeros((filters, input_len));
        let scale = 1.0 / (self.size as f32);

        for f in 0..filters {
            for i in 0..out_len {
                let start = i * self.size;
                let g = grad_output[[f, i]] * scale;
                for j in 0..self.size {
                    grad_input[[f, start + j]] += g;
                }
            }
        }
        grad_input
    }
}

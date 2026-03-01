use ndarray::{Array1, Array2, Array3};
use rand_distr::{Distribution, Normal};

pub struct Conv1D {
    pub weights: Array3<f32>, // (filters, in_channels, kernel)
    pub bias: Array1<f32>,    // (filters)
    pub stride: usize,
    last_input: Option<Array2<f32>>, // (in_channels, len)
}

impl Conv1D {
    pub fn new(filters: usize, in_channels: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_channels * kernel_size) as f32).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        let weights =
            Array3::from_shape_fn((filters, in_channels, kernel_size), |_| dist.sample(&mut rng));
        Self {
            weights,
            bias: Array1::zeros(filters),
            stride: 1,
            last_input: None,
        }
    }

    // input shape: (in_channels, len)
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());

        let (filters, in_channels, k_size) = self.weights.dim();
        let in_len = input.ncols();
        let out_len = (in_len - k_size) / self.stride + 1;

        let mut output = Array2::zeros((filters, out_len));
        for f in 0..filters {
            for t in 0..out_len {
                let mut sum = self.bias[f];
                for c in 0..in_channels {
                    for k in 0..k_size {
                        sum += input[[c, t * self.stride + k]] * self.weights[[f, c, k]];
                    }
                }
                output[[f, t]] = sum;
            }
        }
        output
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.last_input.as_ref().expect("forward() before backward()");
        let (filters, in_channels, k_size) = self.weights.dim();

        let mut grad_weights = Array3::zeros(self.weights.dim());
        let mut grad_bias = Array1::zeros(filters);
        let mut grad_input = Array2::zeros(input.dim());

        for f in 0..filters {
            for t in 0..grad_output.ncols() {
                let g = grad_output[[f, t]];
                grad_bias[f] += g;
                for c in 0..in_channels {
                    for k in 0..k_size {
                        let x = input[[c, t * self.stride + k]];
                        grad_weights[[f, c, k]] += x * g;
                        grad_input[[c, t * self.stride + k]] += self.weights[[f, c, k]] * g;
                    }
                }
            }
        }

        grad_weights.mapv_inplace(|x: f32| x.clamp(-1.0, 1.0));

        self.weights -= &(grad_weights * lr);
        self.bias -= &(grad_bias * lr);

        grad_input
    }
}

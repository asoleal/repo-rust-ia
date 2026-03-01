use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};

pub struct Dense {
    pub weights: Array2<f32>, // (in_dim, out_dim)
    pub bias: Array1<f32>,    // (out_dim)
    last_input: Option<Array2<f32>>, // (1, in_dim)
}

impl Dense {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / in_dim as f32).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        let weights = Array2::from_shape_fn((in_dim, out_dim), |_| dist.sample(&mut rng));
        Self {
            weights,
            bias: Array1::zeros(out_dim),
            last_input: None,
        }
    }

    // input: (1, in_dim) -> output: (1, out_dim)
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        let mut out = input.dot(&self.weights);
        // sumarle bias a la única fila
        for j in 0..self.bias.len() {
            out[[0, j]] += self.bias[j];
        }
        out
    }

    // grad_output: (1, out_dim) -> returns grad_input: (1, in_dim)
    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.last_input.as_ref().expect("forward() before backward()");

        let grad_w = input.t().dot(grad_output); // (in_dim, out_dim)
        let grad_b = grad_output.row(0).to_owned(); // (out_dim)
        let grad_in = grad_output.dot(&self.weights.t()); // (1, in_dim)

        self.weights -= &(grad_w * lr);
        self.bias -= &(grad_b * lr);

        grad_in
    }
}

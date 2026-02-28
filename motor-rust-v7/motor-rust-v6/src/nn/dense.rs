use ndarray::{Array1, Array2, Axis};
use serde::{Serialize, Deserialize};
use rand_distr::{Normal, Distribution};

#[derive(Serialize, Deserialize)]
pub struct Dense {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    #[serde(skip)]
    last_input: Option<Array2<f32>>,
}

impl Dense {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (inputs + outputs) as f32).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        Dense {
            weights: Array2::from_shape_fn((inputs, outputs), |_| dist.sample(&mut rng) as f32),
            bias: Array1::zeros(outputs),
            last_input: None,
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        input.dot(&self.weights) + &self.bias
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.last_input.as_ref().unwrap();
        let grad_weights = input.t().dot(grad_output);
        let grad_bias = grad_output.sum_axis(Axis(0));
        let grad_input = grad_output.dot(&self.weights.t());
        
        self.weights -= &(grad_weights * lr);
        self.bias -= &(grad_bias * lr);
        grad_input
    }
}

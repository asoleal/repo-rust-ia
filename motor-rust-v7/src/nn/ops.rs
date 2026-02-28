use ndarray::Array2;

pub fn leaky_relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|val| if val > 0.0 { val } else { val * 0.01 })
}

pub fn leaky_relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|val| if val > 0.0 { 1.0 } else { 0.01 })
}

pub struct MaxPool1D { pub size: usize }

impl MaxPool1D {
    pub fn new(size: usize) -> Self { MaxPool1D { size } }
    
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (filters, len) = input.dim();
        let out_len = len / self.size;
        let mut output = Array2::zeros((filters, out_len));
        
        for f in 0..filters {
            for i in 0..out_len {
                let start = i * self.size;
                let mut max_val = input[[f, start]];
                for j in 1..self.size {
                    let val = input[[f, start + j]];
                    if val > max_val { max_val = val; }
                }
                output[[f, i]] = max_val;
            }
        }
        output
    }
}

pub mod nn;
pub mod data;

use std::fs::File;
use nn::conv1d::Conv1D;
use nn::dense::Dense;
use nn::ops::{leaky_relu, leaky_relu_prime, MaxPool1D};
use ndarray::{Array2};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct ModelV7 { 
    pub conv: Conv1D, 
    pub dense: Dense 
}

#[no_mangle]
pub extern "C" fn predict_motor_status(data_ptr: *const f32) -> i32 {
    let f = match File::open("modelo_v6.json") {
        Ok(file) => file,
        Err(_) => return -1,
    };
    let mut model: ModelV7 = serde_json::from_reader(f).unwrap();
    let pool = MaxPool1D::new(2);

    let raw_data = unsafe { std::slice::from_raw_parts(data_ptr, 128) };
    let input = Array2::from_shape_vec((1, 128), raw_data.to_vec()).unwrap();

    let z1 = model.conv.forward(&input);
    let a1 = leaky_relu(&z1);
    let p1 = pool.forward(&a1);
    let flat = p1.into_shape((1, 744)).unwrap();
    let output = model.dense.forward(&flat);

    if output[[0, 1]] > output[[0, 0]] { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn train_on_sample(data_ptr: *const f32, correct_label: i32) -> f32 {
    let f = match File::open("modelo_v6.json") {
        Ok(file) => file,
        Err(_) => return -1.0,
    };
    let mut model: ModelV7 = serde_json::from_reader(f).unwrap();
    let pool = MaxPool1D::new(2);
    let lr: f32 = 0.0005; 

    let raw_data = unsafe { std::slice::from_raw_parts(data_ptr, 128) };
    let input = Array2::from_shape_vec((1, 128), raw_data.to_vec()).unwrap();
    
    let mut target = Array2::zeros((1, 2));
    if correct_label == 1 { target[[0, 1]] = 1.0; } else { target[[0, 0]] = 1.0; }

    // Forward
    let z1 = model.conv.forward(&input);
    let a1 = leaky_relu(&z1);
    let p1 = pool.forward(&a1);
    let flat = p1.into_shape((1, 744)).unwrap();
    let output = model.dense.forward(&flat);

    // Backward
    let error = &output - &target;
    let loss = error.mapv(|x| x.powi(2)).sum();

    let grad_dense = model.dense.backward(&error, lr);
    
    // Propagación de gradiente simple a la Conv
    let grad_p1 = grad_dense.into_shape((12, 62)).unwrap();
    let mut grad_a1 = Array2::zeros(a1.dim());
    for f in 0..12 {
        for t in 0..62 {
            grad_a1[[f, t * 2]] = grad_p1[[f, t]];
            grad_a1[[f, t * 2 + 1]] = grad_p1[[f, t]];
        }
    }
    let grad_z1 = &grad_a1 * &leaky_relu_prime(&z1);
    model.conv.backward(&grad_z1, lr);

    // Guardar progreso
    let out_file = File::create("modelo_v6.json").unwrap();
    serde_json::to_writer(out_file, &model).unwrap();

    loss
}

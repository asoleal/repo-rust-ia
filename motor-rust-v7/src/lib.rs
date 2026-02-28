pub mod nn;
pub mod data;

use std::fs::File;
use nn::conv1d::Conv1D;
use nn::dense::Dense;
use nn::ops::{leaky_relu, MaxPool1D};
use ndarray::{Array2};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct ModelV7 { 
    pub conv: Conv1D, 
    pub dense: Dense 
}

// Función principal de exportación para C/Python
#[no_mangle]
pub extern "C" fn predict_motor_status(data_ptr: *const f32) -> i32 {
    // 1. Cargar el cerebro (modelo_v6.json)
    let f = match File::open("modelo_v6.json") {
        Ok(file) => file,
        Err(_) => return -1, // Error: No se encontró el archivo del modelo
    };
    let mut model: ModelV7 = serde_json::from_reader(f).unwrap();
    let pool = MaxPool1D::new(2);

    // 2. Mapear los datos que vienen de afuera (128 samples)
    let raw_data = unsafe { std::slice::from_raw_parts(data_ptr, 128) };
    let input = Array2::from_shape_vec((1, 128), raw_data.to_vec()).unwrap();

    // 3. Forward Pass (Idéntico a tu entrenamiento)
    let z1 = model.conv.forward(&input);
    let a1 = leaky_relu(&z1);
    let p1 = pool.forward(&a1);
    
    // El tamaño flat debe ser: filtros(12) * pool_len(62) = 744
    let flat = p1.into_shape((1, 744)).unwrap();
    let output = model.dense.forward(&flat);

    // 4. Decisión
    if output[[0, 1]] > output[[0, 0]] { 1 } else { 0 }
}

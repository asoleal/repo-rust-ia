use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray::{Array2, Axis};
use std::fs::File;
use std::io::{Write, Read};
use serde::{Serialize, Deserialize};

// Imports condicionales para Python
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray, PyArrayMethods};

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Clone)]
pub struct RedBatched {
    pub pesos: Vec<Array2<f64>>,
    pub sesgos: Vec<Array2<f64>>,
    pub lr: f64,
}

impl RedBatched {
    pub fn internal_forward(&self, x: &ArrayView2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut zs = Vec::new();
        let mut activaciones = vec![x.to_owned()];
        let mut a_actual = x.to_owned();

        for (i, (w, b)) in self.pesos.iter().zip(self.sesgos.iter()).enumerate() {
            let z = a_actual.dot(w) + b;
            zs.push(z.clone());
            a_actual = if i < self.pesos.len() - 1 {
                z.mapv(|val| if val > 0.0 { val } else { 0.0 }) // ReLU
            } else {
                z.mapv(|val| 1.0 / (1.0 + (-val).exp())) // Sigmoide
            };
            activaciones.push(a_actual.clone());
        }
        (zs, activaciones)
    }

    pub fn predict_pure(&self, input: &Array2<f64>) -> Array2<f64> {
        let (_, activaciones) = self.internal_forward(&input.view());
        activaciones.last().unwrap().clone()
    }
}

// Bloque de métodos exclusivo para Python
#[cfg(feature = "python")]
#[pymethods]
impl RedBatched {
    #[new]
    pub fn new(arquitectura: Vec<usize>, lr: f64) -> Self {
        let mut pesos = Vec::new();
        let mut sesgos = Vec::new();
        for i in 0..arquitectura.len() - 1 {
            let std_dev = (2.0 / arquitectura[i] as f64).sqrt();
            pesos.push(Array2::random((arquitectura[i], arquitectura[i + 1]), Normal::new(0.0, std_dev).unwrap()));
            sesgos.push(Array2::zeros((1, arquitectura[i + 1])));
        }
        RedBatched { pesos, sesgos, lr }
    }

    pub fn train_batch(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> f64 {
        let x_arr = x.as_array();
        let y_arr = y.as_array();
        let batch_size = x_arr.shape()[0] as f64;
        let (zs, activaciones) = self.internal_forward(&x_arr);
        let delta_salida = &activaciones[activaciones.len() - 1] - &y_arr;
        let loss = delta_salida.mapv(|v| v.powi(2)).mean().unwrap();
        let mut delta = delta_salida;
        for i in (0..self.pesos.len()).rev() {
            let grad_w = activaciones[i].t().dot(&delta) / batch_size;
            let grad_b = delta.sum_axis(Axis(0)).insert_axis(Axis(0)) / batch_size;
            if i > 0 {
                delta = delta.dot(&self.pesos[i].t()) * zs[i-1].mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            }
            self.pesos[i] -= &(grad_w * self.lr);
            self.sesgos[i] -= &(grad_b * self.lr);
        }
        loss
    }

    pub fn predict(&self, input: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>> {
        let input_matrix = input.to_owned_array();
        let output = self.predict_pure(&input_matrix);
        Python::with_gil(|py| {
            output.to_pyarray(py).to_owned()
        })
    }
}

// Métodos comunes que no dependen de PyO3
impl RedBatched {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let red: Self = serde_json::from_str(&contents).unwrap();
        Ok(red)
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn motor_batches(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RedBatched>()?;
    Ok(())
}

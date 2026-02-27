use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;

#[cfg(feature = "python")]
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Clone)]
pub struct RedNativa {
    pub pesos: Vec<Array2<f64>>,
    pub sesgos: Vec<Array2<f64>>,
    pub lr: f64,
}

impl RedNativa {
    pub fn new_rust(arquitectura: Vec<usize>, lr: f64) -> Self {
        let mut pesos = Vec::new();
        let mut sesgos = Vec::new();
        for i in 0..arquitectura.len() - 1 {
            let std_dev = (2.0 / arquitectura[i] as f64).sqrt();
            pesos.push(Array2::random(
                (arquitectura[i], arquitectura[i + 1]),
                Normal::new(0.0, std_dev).unwrap(),
            ));
            sesgos.push(Array2::zeros((1, arquitectura[i + 1])));
        }
        RedNativa { pesos, sesgos, lr }
    }

    pub fn predict_native(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut a_actual = x.clone();
        for (i, (w, b)) in self.pesos.iter().zip(self.sesgos.iter()).enumerate() {
            let z = a_actual.dot(w) + b;
            a_actual = if i < self.pesos.len() - 1 {
                z.mapv(|val| if val > 0.0 { val } else { 0.0 }) // ReLU
            } else {
                z.mapv(|val| 1.0 / (1.0 + (-val).exp())) // Sigmoide
            };
        }
        a_actual
    }

    pub fn train_native(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> f64 {
        let batch_size = x.shape()[0] as f64;
        let mut zs = Vec::new();
        let mut activaciones = vec![x.clone()];
        let mut a_actual = x.clone();

        for (i, (w, b)) in self.pesos.iter().zip(self.sesgos.iter()).enumerate() {
            let z = a_actual.dot(w) + b;
            zs.push(z.clone());
            a_actual = if i < self.pesos.len() - 1 {
                z.mapv(|val| if val > 0.0 { val } else { 0.0 })
            } else {
                z.mapv(|val| 1.0 / (1.0 + (-val).exp()))
            };
            activaciones.push(a_actual.clone());
        }

        let delta_salida = &activaciones[activaciones.len() - 1] - y;
        let loss = delta_salida.mapv(|v| v.powi(2)).mean().unwrap();
        let mut delta = delta_salida;

        for i in (0..self.pesos.len()).rev() {
            let grad_w = activaciones[i].t().dot(&delta) / batch_size;
            let grad_b = delta.sum_axis(Axis(0)).insert_axis(Axis(0)) / batch_size;
            if i > 0 {
                delta = delta.dot(&self.pesos[i].t())
                    * zs[i - 1].mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            }
            self.pesos[i] -= &(grad_w * self.lr);
            self.sesgos[i] -= &(grad_b * self.lr);
        }
        loss
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl RedNativa {
    #[new]
    pub fn py_new(arquitectura: Vec<usize>, lr: f64) -> Self {
        Self::new_rust(arquitectura, lr)
    }

    pub fn train_batch(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> f64 {
        self.train_native(&x.to_owned_array(), &y.to_owned_array())
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn motor_rust_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RedNativa>()?;
    Ok(())
}

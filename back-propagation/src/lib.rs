use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct RedDinamica {
    pub pesos: Vec<Array2<f64>>,
    pub sesgos: Vec<Array2<f64>>,
    #[pyo3(get)]
    pub learning_rate: f64,
}

#[pymethods]
impl RedDinamica {
    #[new]
    pub fn new(arquitectura: Vec<usize>, learning_rate: f64) -> Self {
        let mut pesos = Vec::new();
        let mut sesgos = Vec::new();

        for i in 0..arquitectura.len() - 1 {
            let std_dev = (2.0 / arquitectura[i] as f64).sqrt();
            pesos.push(Array2::random(
                (arquitectura[i], arquitectura[i + 1]),
                Normal::new(0.0, std_dev).unwrap()
            ));
            sesgos.push(Array2::zeros((1, arquitectura[i + 1])));
        }

        RedDinamica { pesos, sesgos, learning_rate }
    }

    pub fn forward_py(&self, py: Python, x: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>> {
        let input = x.as_array().to_owned();
        let (_, activations) = self.feed_forward(&input);
        activations.last().unwrap().to_pyarray_bound(py).unbind()
    }

    pub fn train_py(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) {
        let x_arr = x.as_array().to_owned();
        let y_arr = y.as_array().to_owned();
        self.train(&x_arr, &y_arr);
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

impl RedDinamica {
    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|a| if a > 0.0 { a } else { 0.0 })
    }

    fn relu_derivada(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|a| if a > 0.0 { 1.0 } else { 0.0 })
    }

    fn sigmoide(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|a| 1.0 / (1.0 + (-a).exp()))
    }

    fn feed_forward(&self, x: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut zs = Vec::new();
        let mut activaciones = vec![x.clone()];
        let mut a_actual = x.clone();

        for (i, (w, b)) in self.pesos.iter().zip(self.sesgos.iter()).enumerate() {
            let z = a_actual.dot(w) + b;
            zs.push(z.clone());
            a_actual = if i < self.pesos.len() - 1 {
                Self::relu(&z)
            } else {
                Self::sigmoide(&z)
            };
            activaciones.push(a_actual.clone());
        }
        (zs, activaciones)
    }

    pub fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>) {
        let (zs, activaciones) = self.feed_forward(x);
        let mut deltas: Vec<Array2<f64>> = Vec::new();
        let n_capas = self.pesos.len();

        // Error en salida
        let delta_salida = &activaciones[activaciones.len() - 1] - y;
        deltas.push(delta_salida);

        // Retropropagación
        for i in (0..n_capas - 1).rev() {
            let d_back = deltas.last().unwrap().dot(&self.pesos[i + 1].t()) * Self::relu_derivada(&zs[i]);
            deltas.push(d_back);
        }
        deltas.reverse();

        // Actualización
        for i in 0..n_capas {
            let grad_w = activaciones[i].t().dot(&deltas[i]);
            let grad_b = deltas[i].sum_axis(Axis(0)).insert_axis(Axis(0));
            self.pesos[i] = &self.pesos[i] - &(grad_w * self.learning_rate);
            self.sesgos[i] = &self.sesgos[i] - &(grad_b * self.learning_rate);
        }
    }
}

#[pymodule]
fn mi_motor_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RedDinamica>()?;
    Ok(())
}

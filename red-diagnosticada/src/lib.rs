use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};

#[pyclass]
#[derive(Clone)]
pub struct TrainingStats {
    #[pyo3(get)]
    pub loss: f64,
    #[pyo3(get)]
    pub alive_neurons_ratio: f64,
}

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct RedDinamica {
    pub pesos: Vec<Array2<f64>>,
    pub sesgos: Vec<Array2<f64>>,
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
            pesos.push(Array2::random((arquitectura[i], arquitectura[i + 1]), Normal::new(0.0, std_dev).unwrap()));
            sesgos.push(Array2::zeros((1, arquitectura[i + 1])));
        }
        RedDinamica { pesos, sesgos, learning_rate }
    }

    pub fn train_diagnostico(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> TrainingStats {
        let x_arr = x.as_array().to_owned();
        let y_arr = y.as_array().to_owned();
        
        let (zs, activaciones) = self.feed_forward(&x_arr);
        
        // Análisis de Neuronas Muertas (Basado en ReLU en capas ocultas)
        let mut total_neurons = 0.0;
        let mut alive_neurons = 0.0;
        for z in &zs {
            total_neurons += z.len() as f64;
            alive_neurons += z.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 }).sum();
        }

        // Backpropagation manual (Referencia: Sarkar, 2026)
        let n_capas = self.pesos.len();
        let mut deltas = Vec::new();
        let delta_salida = &activaciones[activaciones.len() - 1] - &y_arr;
        let loss = delta_salida.mapv(|a| a.powi(2)).mean().unwrap_or(0.0);
        
        deltas.push(delta_salida);
        for i in (0..n_capas - 1).rev() {
            let d_back = deltas.last().unwrap().dot(&self.pesos[i + 1].t()) * zs[i].mapv(|a| if a > 0.0 { 1.0 } else { 0.0 });
            deltas.push(d_back);
        }
        deltas.reverse();

        for i in 0..n_capas {
            let grad_w = activaciones[i].t().dot(&deltas[i]);
            let grad_b = deltas[i].sum_axis(Axis(0)).insert_axis(Axis(0));
            self.pesos[i] -= &(grad_w * self.learning_rate);
            self.sesgos[i] -= &(grad_b * self.learning_rate);
        }

        TrainingStats { loss, alive_neurons_ratio: alive_neurons / total_neurons }
    }

    pub fn forward_py(&self, py: Python, x: PyReadonlyArray2<f64>) -> Py<PyArray2<f64>> {
        let input = x.as_array().to_owned();
        let (_, activations) = self.feed_forward(&input);
        activations.last().unwrap().to_pyarray_bound(py).unbind()
    }
}

impl RedDinamica {
    fn feed_forward(&self, x: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut zs = Vec::new();
        let mut activaciones = vec![x.clone()];
        let mut a_actual = x.clone();

        for (i, (w, b)) in self.pesos.iter().zip(self.sesgos.iter()).enumerate() {
            let z = a_actual.dot(w) + b;
            zs.push(z.clone());
            a_actual = if i < self.pesos.len() - 1 {
                z.mapv(|a| if a > 0.0 { a } else { 0.0 }) // ReLU
            } else {
                z.mapv(|a| 1.0 / (1.0 + (-a).exp())) // Sigmoide
            };
            activaciones.push(a_actual.clone());
        }
        (zs, activaciones)
    }
}

#[pymodule]
fn mi_motor_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RedDinamica>()?;
    m.add_class::<TrainingStats>()?;
    Ok(())
}

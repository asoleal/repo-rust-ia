use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct RegresionLogistica {
    tasa_aprendizaje: f64,
    iteraciones: usize,
    #[pyo3(get)]
    pub peso: f64,
    #[pyo3(get)]
    pub sesgo: f64,
}

// Función matemática para clasificación (0 a 1)
fn sigmoide(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[pymethods]
impl RegresionLogistica {
    #[new]
    pub fn new(tasa_aprendizaje: f64, iteraciones: usize) -> Self {
        Self {
            tasa_aprendizaje,
            iteraciones,
            peso: 0.0,
            sesgo: 0.0,
        }
    }

    pub fn entrenar(&mut self, x: Vec<f64>, y: Vec<f64>) {
        let n = x.len() as f64;

        for _ in 0..self.iteraciones {
            let (dw_sum, db_sum) = (0..x.len())
                .into_par_iter()
                .map(|i| {
                    // CAMBIO CLAVE: Aplicamos sigmoide a la combinación lineal
                    let z = self.peso * x[i] + self.sesgo;
                    let y_pred = sigmoide(z);
                    let error = y_pred - y[i];
                    (error * x[i], error)
                })
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

            self.peso -= self.tasa_aprendizaje * (dw_sum / n);
            self.sesgo -= self.tasa_aprendizaje * (db_sum / n);
        }
    }

    // Cambiamos el nombre para que sea más claro en Python
    pub fn predecir_probabilidad(&self, x: Vec<f64>) -> Vec<f64> {
        x.into_iter()
            .map(|val| sigmoide(self.peso * val + self.sesgo))
            .collect()
    }
}

#[pymodule]
fn regresion_lineal(_py: Python, m: &PyModule) -> PyResult<()> {
    // Ahora el nombre aquí coincide con el struct de arriba
    m.add_class::<RegresionLogistica>()?;
    Ok(())
}

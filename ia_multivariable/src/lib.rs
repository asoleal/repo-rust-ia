use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct IAMultiple {
    tasa_aprendizaje: f64,
    iteraciones: usize,
    #[pyo3(get)]
    pub pesos: Vec<f64>, // Un peso para cada entrada (ej: longitud, enlaces, palabras clave)
    #[pyo3(get)]
    pub sesgo: f64,
}

fn sigmoide(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[pymethods]
impl IAMultiple {
    #[new]
    pub fn new(tasa_aprendizaje: f64, iteraciones: usize, num_caracteristicas: usize) -> Self {
        Self {
            tasa_aprendizaje,
            iteraciones,
            // Inicializamos todos los pesos en 0.0
            pesos: vec![0.0; num_caracteristicas],
            sesgo: 0.0,
        }
    }

    // X ahora es una lista de listas (Vec de Vec)
    pub fn entrenar(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        let n = x.len() as f64;
        let num_features = self.pesos.len();

        for _ in 0..self.iteraciones {
            // Calculamos el error global de forma paralela
            let (dw_sum, db_sum) = (0..x.len())
                .into_par_iter()
                .map(|i| {
                    // Producto punto: sumatoria de (peso * x) + sesgo
                    let mut z = self.sesgo;
                    for j in 0..num_features {
                        z += self.pesos[j] * x[i][j];
                    }

                    let y_pred = sigmoide(z);
                    let error = y_pred - y[i];

                    // Calculamos el gradiente para cada peso de este ejemplo
                    let mut grad_pesos = vec![0.0; num_features];
                    for j in 0..num_features {
                        grad_pesos[j] = error * x[i][j];
                    }

                    (grad_pesos, error)
                })
                .reduce(
                    || (vec![0.0; num_features], 0.0),
                    |mut a, b| {
                        for j in 0..num_features { a.0[j] += b.0[j]; }
                        (a.0, a.1 + b.1)
                    }
                );

            // Actualizamos todos los pesos
            for j in 0..num_features {
                self.pesos[j] -= self.tasa_aprendizaje * (dw_sum[j] / n);
            }
            self.sesgo -= self.tasa_aprendizaje * (db_sum / n);
        }
    }
    pub fn predecir(&self, x: Vec<Vec<f64>>) -> Vec<f64> {
        x.into_iter()
            .map(|punto| {
                let mut z = self.sesgo;
                for j in 0..self.pesos.len() {
                    z += self.pesos[j] * punto[j];
                }
                sigmoide(z)
            })
            .collect()
    }
}

#[pymodule]
fn regresion_lineal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IAMultiple>()?;
    Ok(())
}

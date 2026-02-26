use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

#[pyclass]
// Esta línea es la "magia" que permite a Serde manejar tu IA
#[derive(Serialize, Deserialize)]
pub struct IAMultiple {
    pub tasa_aprendizaje: f64,
    pub iteraciones: usize,
    #[pyo3(get)]
    pub pesos: Vec<f64>,
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
    pub fn guardar(&self, ruta: String) -> PyResult<()> {
        // 1. Convertimos la estructura actual a un String en formato JSON
        let json = serde_json::to_string(self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        // 2. Creamos el archivo y escribimos el contenido
        let mut archivo = File::create(ruta)?;
        archivo.write_all(json.as_bytes())?;

        println!("✅ Modelo guardado exitosamente.");
        Ok(())
    }
    #[staticmethod]
    pub fn cargar(ruta: String) -> PyResult<Self> {
        // 1. Abrimos el archivo en modo lectura
        let mut archivo = File::open(ruta)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        // 2. Leemos todo el contenido a un String
        let mut contenido = String::new();
        archivo.read_to_string(&mut contenido)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        // 3. Convertimos el JSON de vuelta a la estructura de Rust
        let modelo: Self = serde_json::from_str(&contenido)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        println!("📂 Modelo cargado exitosamente.");
        Ok(modelo)
    }
}

#[pymodule]
fn regresion_lineal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IAMultiple>()?;
    Ok(())
}

use ndarray::prelude::*;
use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct IAMultiple {
    pub tasa_aprendizaje: f64,
    pub iteraciones: usize,
    // Cambio: De Vec<f64> a Array1<f64>
    pub pesos: Array1<f64>,
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
            pesos: Array1::zeros(num_caracteristicas),
            sesgo: 0.0,
        }
    }
    pub fn entrenar(&mut self, x_py: Vec<Vec<f64>>, y_py: Vec<f64>) {
        let filas = x_py.len();
        let cols = x_py[0].len();

        // Convertimos los datos de Python a matrices de ndarray
        let mut x_flat = Vec::with_capacity(filas * cols);
        for fila in x_py { x_flat.extend(fila); }
        let x = Array2::from_shape_vec((filas, cols), x_flat).unwrap();
        let y = Array1::from_vec(y_py);

        let n = filas as f64;

        for _ in 0..self.iteraciones {
            // --- EL CORAZÓN DE LA IA MATRICIAL ---
            // 1. Predicción: z = (X · pesos) + sesgo
            let z = x.dot(&self.pesos) + self.sesgo;

            // 2. Aplicamos sigmoide a todo el vector de golpe
            let y_pred = z.mapv(sigmoide);

            // 3. Cálculo del error
            let error = &y_pred - &y;

            // 4. Gradientes: dw = (X_transpuesta · error) / n
            let dw = x.t().dot(&error) / n;
            let db = error.sum() / n;

            // 5. Actualización de parámetros (Operación vectorial)
            self.pesos -= &(dw * self.tasa_aprendizaje);
            self.sesgo -= db * self.tasa_aprendizaje;
        }
        println!("🚀 Entrenamiento matricial completado.");
    }

    pub fn predecir(&self, x_py: Vec<Vec<f64>>) -> Vec<f64> {
        let filas = x_py.len();
        let cols = x_py[0].len();
        let mut x_flat = Vec::with_capacity(filas * cols);
        for fila in x_py { x_flat.extend(fila); }

        let x = Array2::from_shape_vec((filas, cols), x_flat).unwrap();

        // Predicción rápida: sigmoide(X · pesos + sesgo)
        let z = x.dot(&self.pesos) + self.sesgo;
        z.mapv(sigmoide).to_vec()
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
    #[getter]
    pub fn pesos(&self) -> Vec<f64> {
        self.pesos.to_vec() // Convertimos la matriz a una lista que Python entiende
    }
    #[getter]
    pub fn sesgo(&self) -> f64 {
        self.sesgo
    }
}

#[pymodule]
fn regresion_lineal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IAMultiple>()?;
    Ok(())
}

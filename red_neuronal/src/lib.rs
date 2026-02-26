use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct RedNeuronal {
    pub tasa_aprendizaje: f64,
    pub iteraciones: usize,
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array1<f64>,
    pub b2: f64,
}

fn sigmoide(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn sigmoide_derivada(a: f64) -> f64 {
    a * (1.0 - a)
}

#[pymethods]
impl RedNeuronal {
    #[new]
    pub fn new(tasa_aprendizaje: f64, iteraciones: usize, entradas: usize, ocultas: usize) -> Self {
        let dist = Normal::new(0.0, 0.1).unwrap();
        Self {
            tasa_aprendizaje,
            iteraciones,
            w1: Array2::random((entradas, ocultas), dist),
            b1: Array1::zeros(ocultas),
            w2: Array1::random(ocultas, dist),
            b2: 0.0,
        }
    }

    pub fn entrenar(&mut self, x_py: Vec<Vec<f64>>, y_py: Vec<f64>) {
        let filas = x_py.len();
        let cols = x_py[0].len();
        let mut x_flat = Vec::with_capacity(filas * cols);
        for fila in x_py { x_flat.extend(fila); }

        let x = Array2::from_shape_vec((filas, cols), x_flat).unwrap();
        let y = Array1::from_vec(y_py);
        let n = filas as f64;

        for _ in 0..self.iteraciones {
            // --- 1. FORWARD PASS ---
            // Capa oculta
            let z1 = x.dot(&self.w1) + &self.b1;
            let a1 = z1.mapv(sigmoide);

            // Capa de salida
            let z2 = a1.dot(&self.w2) + self.b2;
            let a2 = z2.mapv(sigmoide);

            // --- 2. BACKPROPAGATION ---
            // Error en salida (a2 - y)
            let d_z2 = &a2 - &y;

            // Gradientes para W2 y B2
            let d_w2 = a1.t().dot(&d_z2) / n;
            let d_b2 = d_z2.sum() / n;

            // Error en capa oculta (Regla de la cadena)
            // Proyectamos el error de salida hacia atrás a través de W2
            let d_a1 = d_z2.insert_axis(Axis(1)).dot(&self.w2.clone().insert_axis(Axis(0)));
            let d_z1 = d_a1 * a1.mapv(sigmoide_derivada);

            // Gradientes para W1 y B1
            let d_w1 = x.t().dot(&d_z1) / n;
            let d_b1 = d_z1.sum_axis(Axis(0)) / n;

            // --- 3. ACTUALIZACIÓN DE PESOS ---
            self.w1 -= &(d_w1 * self.tasa_aprendizaje);
            self.b1 -= &(d_b1 * self.tasa_aprendizaje);
            self.w2 -= &(d_w2 * self.tasa_aprendizaje);
            self.b2 -= d_b2 * self.tasa_aprendizaje;
        }
        println!("🧠 Red Neuronal entrenada con éxito.");
    }

    pub fn predecir(&self, x_py: Vec<Vec<f64>>) -> Vec<f64> {
        let filas = x_py.len();
        let cols = x_py[0].len();
        let mut x_flat = Vec::with_capacity(filas * cols);
        for fila in x_py { x_flat.extend(fila); }
        let x = Array2::from_shape_vec((filas, cols), x_flat).unwrap();

        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(sigmoide);
        let z2 = a1.dot(&self.w2) + self.b2;
        z2.mapv(sigmoide).to_vec()
    }

    pub fn guardar(&self, ruta: String) -> PyResult<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut archivo = File::create(ruta)?;
        archivo.write_all(json.as_bytes())?;
        Ok(())
    }

    #[staticmethod]
    pub fn cargar(ruta: String) -> PyResult<Self> {
        let mut archivo = File::open(ruta)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut contenido = String::new();
        archivo.read_to_string(&mut contenido)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let modelo: Self = serde_json::from_str(&contenido)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(modelo)
    }
    // --- AÑADE ESTO AL FINAL DEL impl RedNeuronal ---

    #[getter]
    pub fn tasa_aprendizaje(&self) -> f64 { self.tasa_aprendizaje }

    #[getter]
    pub fn w1(&self) -> Vec<Vec<f64>> {
        // Convertimos la matriz de Rust a una lista de listas de Python
        self.w1.axis_iter(Axis(0))
            .map(|row| row.to_vec())
            .collect()
    }

    #[getter]
    pub fn w2(&self) -> Vec<f64> {
        self.w2.to_vec()
    }
}

#[pymodule]
fn red_neuronal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RedNeuronal>()?;
    Ok(())
}

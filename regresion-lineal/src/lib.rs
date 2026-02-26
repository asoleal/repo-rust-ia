use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct RegresionLineal {
    // Configuración que nosotros le damos:
    tasa_aprendizaje: f64,
    iteraciones: usize,

    // Agregamos #[pyo3(get)] para poder leer estos valores desde Python
    #[pyo3(get)]
    pub peso: f64,
    #[pyo3(get)]
    pub sesgo: f64,
}

#[pymethods]
impl RegresionLineal {

    // 1. El Constructor (Crea el modelo en blanco)
    #[new]
    pub fn new(tasa_aprendizaje: f64, iteraciones: usize) -> Self {
        Self {
            tasa_aprendizaje: tasa_aprendizaje,
            iteraciones: iteraciones,
            peso: 0.0,
            sesgo: 0.0
        }
    }

    // 2. La función de Entrenamiento (Descenso del Gradiente)
    pub fn entrenar(&mut self, x: Vec<f64>, y: Vec<f64>) {
        let n = x.len() as f64;

        for _ in 0..self.iteraciones {
            // Usamos Rayon para procesar el rango de índices en paralelo
            // .map() calcula el error de cada punto y .reduce() suma todos los resultados
            let (dw_sum, db_sum) = (0..x.len())
                .into_par_iter()
                .map(|i| {
                    let y_pred = self.peso * x[i] + self.sesgo;
                    let error = y_pred - y[i];
                    (error * x[i], error) // Devolvemos un par (dw, db) para este dato
                })
                // Sumamos todos los pares calculados en los diferentes núcleos
                .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

            // El resto sigue igual, pero usando los resultados de la suma paralela
            let dw = dw_sum / n;
            let db = db_sum / n;

            self.peso -= self.tasa_aprendizaje * dw;
            self.sesgo -= self.tasa_aprendizaje * db;
        }
    }

    // 3. La función de Predicción (Usar lo aprendido)
    // CORRECCIÓN: Faltaba cerrar el paréntesis de la 'x' aquí abajo
    pub fn predecir(&self, x: Vec<f64>) -> Vec<f64> {
        let mut predicciones = Vec::new();

        for i in 0..x.len() {
            let y_pred = self.peso * x[i] + self.sesgo;
            predicciones.push(y_pred);
        }

        predicciones
    }
}

#[pymodule]
fn regresion_lineal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RegresionLineal>()?;
    Ok(())
}

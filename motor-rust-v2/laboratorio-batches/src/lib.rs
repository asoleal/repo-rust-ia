use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};

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
            pesos.push(Array2::random((arquitectura[i], arquitectura[i + 1]), Normal::new(0.0, std_dev).unwrap()));
            sesgos.push(Array2::zeros((1, arquitectura[i + 1])));
        }
        RedNativa { pesos, sesgos, lr }
    }

    pub fn train_hogwild(&mut self, x: ArrayView2<f64>, y: ArrayView2<f64>) -> f64 {
        let batch_size = x.shape()[0] as f64;
        let mut zs = Vec::new();
        let mut activaciones = vec![x.to_owned()];
        let mut a_actual = x.to_owned();

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

        let delta_salida = &activaciones[activaciones.len() - 1] - &y;
        let mut delta = delta_salida;

        for i in (0..self.pesos.len()).rev() {
            let grad_w = activaciones[i].t().dot(&delta) / batch_size;
            let grad_b = delta.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)) / batch_size;
            if i > 0 {
                delta = delta.dot(&self.pesos[i].t()) * zs[i-1].mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            }
            self.pesos[i] -= &(grad_w * self.lr);
            self.sesgos[i] -= &(grad_b * self.lr);
        }
        0.0
    }

    pub fn guardar(&self, ruta: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&self)?;
        std::fs::write(ruta, json)?;
        Ok(())
    }

    pub fn cargar(ruta: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contenido = std::fs::read_to_string(ruta)?;
        let red: RedNativa = serde_json::from_str(&contenido)?;
        Ok(red)
    }
}

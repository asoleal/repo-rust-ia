use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};

pub mod mnist_loader;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum Activacion { ReLU, LeakyReLU, Sigmoide, Softmax }

#[derive(Serialize, Deserialize, Clone)]
pub struct CapaDensa {
    pub pesos: Array2<f64>,
    pub sesgos: Array2<f64>,
    pub activacion: Activacion,
    #[serde(skip)] pub m_w: Array2<f64>, // Momento 1 para Adam
    #[serde(skip)] pub v_w: Array2<f64>, // Momento 2 para Adam
    #[serde(skip)] pub ultima_entrada: Option<Array2<f64>>,
    #[serde(skip)] pub ultima_z: Option<Array2<f64>>,
}

impl CapaDensa {
    pub fn new(entradas: usize, salidas: usize, activacion: Activacion) -> Self {
        let std_dev = (2.0 / entradas as f64).sqrt();
        Self {
            pesos: Array2::random((entradas, salidas), Normal::new(0.0, std_dev).unwrap()),
            sesgos: Array2::zeros((1, salidas)),
            activacion,
            m_w: Array2::zeros((entradas, salidas)),
            v_w: Array2::zeros((entradas, salidas)),
            ultima_entrada: None,
            ultima_z: None,
        }
    }

    fn aplicar_activacion(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activacion {
            Activacion::ReLU => z.mapv(|v| v.max(0.0)),
            Activacion::LeakyReLU => z.mapv(|v| if v > 0.0 { v } else { 0.01 * v }),
            Activacion::Sigmoide => z.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activacion::Softmax => {
                let mut res = Array2::zeros(z.raw_dim());
                for (i, fila) in z.genrows().into_iter().enumerate() {
                    let max = fila.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                    let exps = fila.mapv(|x| (x - max).exp());
                    let sum = exps.sum();
                    res.row_mut(i).assign(&(exps / sum));
                }
                res
            }
        }
    }
}

pub struct RedModular {
    pub capas: Vec<CapaDensa>,
    pub lr: f64,
    pub t: f64, // Contador de pasos para Adam
}

impl RedModular {
    pub fn new(lr: f64) -> Self {
        Self { capas: Vec::new(), lr, t: 0.0 }
    }

    pub fn agregar_capa(&mut self, capa: CapaDensa) { self.capas.push(capa); }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut act = x.clone();
        for capa in &mut self.capas {
            capa.ultima_entrada = Some(act.clone());
            let z = act.dot(&capa.pesos) + &capa.sesgos;
            capa.ultima_z = Some(z.clone());
            act = capa.aplicar_activacion(&z);
        }
        act
    }

    pub fn backward_adam(&mut self, grad_salida: &Array2<f64>) {
        self.t += 1.0;
        let (beta1, beta2, eps) = (0.9, 0.999, 1e-8);
        let mut grad_actual = grad_salida.clone();

        for capa in self.capas.iter_mut().rev() {
            let entrada = capa.ultima_entrada.as_ref().unwrap();
            let m = entrada.nrows() as f64;
            
            // Si es Softmax + CrossEntropy, delta es simplemente pred - y (ya viene en grad_salida)
            let delta = grad_actual.clone(); 
            let grad_w = entrada.t().dot(&delta) / m;
            let grad_b = delta.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;

            grad_actual = delta.dot(&capa.pesos.t());

            // --- Lógica Adam ---
            capa.m_w = &capa.m_w * beta1 + &grad_w * (1.0 - beta1);
            capa.v_w = &capa.v_w * beta2 + &grad_w.mapv(|x| x.powi(2)) * (1.0 - beta2);

            let m_hat = &capa.m_w / (1.0 - beta1.powf(self.t));
            let v_hat = &capa.v_w / (1.0 - beta2.powf(self.t));

            capa.pesos -= &((m_hat * self.lr) / (v_hat.mapv(|x| x.sqrt()) + eps));
            capa.sesgos -= &(&grad_b * self.lr);
        }
    }

    pub fn guardar(&self, ruta: &str) -> serde_json::Result<()> {
        let json = serde_json::to_string_pretty(&self.capas).unwrap();
        let mut file = File::create(ruta).unwrap();
        file.write_all(json.as_bytes()).unwrap();
        Ok(())
    }

    pub fn cargar(ruta: &str) -> Self {
        let mut file = File::open(ruta).unwrap();
        let mut contenido = String::new();
        file.read_to_string(&mut contenido).unwrap();
        let capas: Vec<CapaDensa> = serde_json::from_str(&contenido).unwrap();
        Self { capas, lr: 0.001, t: 0.0 }
    }
}

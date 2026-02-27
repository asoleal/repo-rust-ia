use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum Activacion { ReLU, LeakyReLU, Sigmoide }

#[derive(Serialize, Deserialize, Clone)]
pub struct CapaDensa {
    pub pesos: Array2<f64>,
    pub sesgos: Array2<f64>,
    pub activacion: Activacion,
    #[serde(skip)]
    pub ultima_entrada: Option<Array2<f64>>,
    #[serde(skip)]
    pub ultima_z: Option<Array2<f64>>,
}

// Implementación de Send + Sync manual para permitir Hogwild!
unsafe impl Send for CapaDensa {}
unsafe impl Sync for CapaDensa {}

impl CapaDensa {
    pub fn new(entradas: usize, salidas: usize, activacion: Activacion) -> Self {
        let gain = match activacion {
            Activacion::Sigmoide => 1.0,
            _ => 2.0_f64.sqrt(),
        };
        let std_dev = gain / (entradas as f64).sqrt();
        Self {
            pesos: Array2::random((entradas, salidas), Normal::new(0.0, std_dev).unwrap()),
            sesgos: Array2::zeros((1, salidas)),
            activacion,
            ultima_entrada: None,
            ultima_z: None,
        }
    }

    fn aplicar_activacion(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activacion {
            Activacion::ReLU => z.mapv(|v| if v > 0.0 { v } else { 0.0 }),
            Activacion::LeakyReLU => z.mapv(|v| if v > 0.0 { v } else { 0.01 * v }),
            Activacion::Sigmoide => z.mapv(|v| 1.0 / (1.0 + (-v).exp())),
        }
    }

    fn derivada_activacion(&self, z: &Array2<f64>) -> Array2<f64> {
        match self.activacion {
            Activacion::ReLU => z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activacion::LeakyReLU => z.mapv(|v| if v > 0.0 { 1.0 } else { 0.01 }),
            Activacion::Sigmoide => {
                let s = z.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                &s * &(1.0 - &s)
            },
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RedModular {
    pub capas: Vec<CapaDensa>,
    pub lr: f64,
}

impl RedModular {
    pub fn new(lr: f64) -> Self {
        Self { capas: Vec::new(), lr }
    }

    pub fn agregar_capa(&mut self, capa: CapaDensa) {
        self.capas.push(capa);
    }

    // Forward modificado para no mutar el objeto global durante el test
    pub fn forward_thread_safe(&self, x: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut activacion_actual = x.clone();
        let mut entradas = Vec::new();
        let mut zs = Vec::new();

        for capa in &self.capas {
            entradas.push(activacion_actual.clone());
            let z = activacion_actual.dot(&capa.pesos) + &capa.sesgos;
            zs.push(z.clone());
            activacion_actual = capa.aplicar_activacion(&z);
        }
        (activacion_actual, entradas, zs)
    }

    // El núcleo de Hogwild!: Actualización sin bloqueos usando punteros crudos
    pub unsafe fn update_hogwild(&self, grad_salida: &Array2<f64>, entradas: &[Array2<f64>], zs: &[Array2<f64>]) {
        let mut grad_actual = grad_salida.clone();
        
        for (i, capa) in self.capas.iter().enumerate().rev() {
            let z = &zs[i];
            let entrada = &entradas[i];
            let m = entrada.nrows() as f64;

            let delta = &grad_actual * &capa.derivada_activacion(z);
            let grad_w = entrada.t().dot(&delta) / m;
            let grad_b = delta.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;

            grad_actual = delta.dot(&capa.pesos.t());

            // ACTUALIZACIÓN ATÓMICA (HOGWILD): Modificamos los pesos directamente en memoria
            let pesos_ptr = capa.pesos.as_ptr() as *mut f64;
            for j in 0..capa.pesos.len() {
                *pesos_ptr.add(j) -= grad_w.as_slice().unwrap()[j] * self.lr;
            }
            let sesgos_ptr = capa.sesgos.as_ptr() as *mut f64;
            for j in 0..capa.sesgos.len() {
                *sesgos_ptr.add(j) -= grad_b.as_slice().unwrap()[j] * self.lr;
            }
        }
    }
}

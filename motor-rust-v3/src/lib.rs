use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};

pub mod mnist_loader;

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

impl CapaDensa {
    pub fn new(entradas: usize, salidas: usize, activacion: Activacion) -> Self {
        let gain = match activacion {
            Activacion::Sigmoide => 1.0,
            _ => 2.0_f64.sqrt(), // He initialization para ReLU/LeakyReLU
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

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut activacion_actual = x.clone();
        for capa in &mut self.capas {
            capa.ultima_entrada = Some(activacion_actual.clone());
            let z = activacion_actual.dot(&capa.pesos) + &capa.sesgos;
            capa.ultima_z = Some(z.clone());
            activacion_actual = capa.aplicar_activacion(&z);
        }
        activacion_actual
    }

    pub fn backward(&mut self, grad_salida: &Array2<f64>) {
        let mut grad_actual = grad_salida.clone();
        for capa in self.capas.iter_mut().rev() {
            let z = capa.ultima_z.as_ref().unwrap();
            let entrada = capa.ultima_entrada.as_ref().unwrap();
            let m = entrada.nrows() as f64;

            let delta = &grad_actual * &capa.derivada_activacion(z);
            let grad_w = entrada.t().dot(&delta) / m;
            let grad_b = delta.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;

            grad_actual = delta.dot(&capa.pesos.t());

            capa.pesos = &capa.pesos - &(grad_w * self.lr);
            capa.sesgos = &capa.sesgos - &(grad_b * self.lr);
        }
    }

    pub fn guardar(&self, ruta: &str) -> serde_json::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(ruta).expect("Error al crear archivo");
        file.write_all(json.as_bytes()).expect("Error al escribir");
        Ok(())
    }

    pub fn cargar(ruta: &str) -> serde_json::Result<Self> {
        let mut file = File::open(ruta).expect("Error al abrir archivo");
        let mut contenido = String::new();
        file.read_to_string(&mut contenido).expect("Error al leer");
        serde_json::from_str(&contenido)
    }
}

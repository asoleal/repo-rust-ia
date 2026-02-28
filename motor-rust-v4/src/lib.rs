use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

pub mod mnist_loader;

pub trait Layer {
    fn forward(&mut self, entrada: &Array4<f64>) -> Array4<f64>;
    fn backward(&mut self, grad_salida: &Array4<f64>) -> Array4<f64>;
    // Para guardar/cargar
    fn get_weights(&self) -> (Option<Array4<f64>>, Option<Array1<f64>>) { (None, None) }
    fn set_weights(&mut self, _w: Array4<f64>, _b: Array1<f64>) {}
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CapaConv2D {
    pub filtros: Array4<f64>,
    pub sesgos: Array1<f64>,
    #[serde(skip)]
    pub entrada_guardada: Option<Array4<f64>>,
    pub lr: f64,
}

impl CapaConv2D {
    pub fn new(n_f: usize, c_in: usize, f_s: usize, lr: f64) -> Self {
        let std_dev = (2.0 / (c_in * f_s * f_s) as f64).sqrt();
        Self {
            filtros: Array4::random((n_f, c_in, f_s, f_s), Normal::new(0.0, std_dev).unwrap()),
            sesgos: Array1::zeros(n_f),
            entrada_guardada: None,
            lr,
        }
    }
}

impl Layer for CapaConv2D {
    fn forward(&mut self, entrada: &Array4<f64>) -> Array4<f64> {
        self.entrada_guardada = Some(entrada.clone());
        let (b, _c_in, h_in, w_in) = entrada.dim();
        let (n_f, _c_f, f_h, f_w) = self.filtros.dim();
        let mut salida = Array4::zeros((b, n_f, h_in - f_h + 1, w_in - f_w + 1));

        salida.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(i, mut out_batch)| {
            for f in 0..n_f {
                for r in 0..out_batch.shape()[1] {
                    for c in 0..out_batch.shape()[2] {
                        let ventana = entrada.slice(s![i, .., r..r+f_h, c..c+f_w]);
                        let suma = (&ventana * &self.filtros.slice(s![f, .., .., ..])).sum();
                        let val = suma + self.sesgos[f];
                        out_batch[[f, r, c]] = if val > 0.0 { val } else { 0.01 * val };
                    }
                }
            }
        });
        salida
    }

    fn backward(&mut self, grad_salida: &Array4<f64>) -> Array4<f64> {
        let entrada = self.entrada_guardada.as_ref().expect("No hay entrada");
        let (b, _c_in, _h_in, _w_in) = entrada.dim();
        let (n_f, _c_f, f_h, f_w) = self.filtros.dim();
        let mut grad_filtros = Array4::zeros(self.filtros.dim());
        let mut grad_entrada = Array4::zeros(entrada.dim());
        let mut grad_sesgos = Array1::zeros(n_f);

        for n in 0..b {
            for f in 0..n_f {
                for r in 0..grad_salida.shape()[2] {
                    for c in 0..grad_salida.shape()[3] {
                        let g = grad_salida[[n, f, r, c]];
                        grad_sesgos[f] += g;
                        let mut seccion_filtros = grad_filtros.slice_mut(s![f, .., .., ..]);
                        seccion_filtros += &(entrada.slice(s![n, .., r..r+f_h, c..c+f_w]).to_owned() * g);
                        let mut seccion_entrada = grad_entrada.slice_mut(s![n, .., r..r+f_h, c..c+f_w]);
                        seccion_entrada += &(self.filtros.slice(s![f, .., .., ..]).to_owned() * g);
                    }
                }
            }
        }
        self.filtros -= &(grad_filtros * self.lr);
        self.sesgos -= &(grad_sesgos * self.lr);
        grad_entrada
    }

    fn get_weights(&self) -> (Option<Array4<f64>>, Option<Array1<f64>>) {
        (Some(self.filtros.clone()), Some(self.sesgos.clone()))
    }
    fn set_weights(&mut self, w: Array4<f64>, b: Array1<f64>) {
        self.filtros = w;
        self.sesgos = b;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CapaPooling { pub size: usize }
impl Layer for CapaPooling {
    fn forward(&mut self, entrada: &Array4<f64>) -> Array4<f64> {
        let (b, c, h, w) = entrada.dim();
        let mut salida = Array4::zeros((b, c, h / self.size, w / self.size));
        for n in 0..b {
            for ch in 0..c {
                for i in 0..(h/self.size) {
                    for j in 0..(w/self.size) {
                        let ventana = entrada.slice(s![n, ch, i*self.size..(i+1)*self.size, j*self.size..(j+1)*self.size]);
                        salida[[n, ch, i, j]] = ventana.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                    }
                }
            }
        }
        salida
    }
    fn backward(&mut self, grad_salida: &Array4<f64>) -> Array4<f64> {
        let (b, c, h_out, w_out) = grad_salida.dim();
        let mut grad_entrada = Array4::zeros((b, c, h_out * self.size, w_out * self.size));
        for n in 0..b {
            for ch in 0..c {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let g = grad_salida[[n, ch, i, j]];
                        grad_entrada.slice_mut(s![n, ch, i*self.size..(i+1)*self.size, j*self.size..(j+1)*self.size]).fill(g / (self.size*self.size) as f64);
                    }
                }
            }
        }
        grad_entrada
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Flatten { pub shape_entrada: Option<(usize, usize, usize, usize)> }
impl Layer for Flatten {
    fn forward(&mut self, entrada: &Array4<f64>) -> Array4<f64> {
        self.shape_entrada = Some(entrada.dim());
        let (b, c, h, w) = entrada.dim();
        entrada.clone().into_shape((b, c * h * w, 1, 1)).unwrap()
    }
    fn backward(&mut self, grad_salida: &Array4<f64>) -> Array4<f64> {
        let (b, c, h, w) = self.shape_entrada.expect("Flatten sin shape");
        grad_salida.clone().into_shape((b, c, h, w)).unwrap()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CapaDensa {
    pub pesos: Array2<f64>,
    pub sesgos: Array2<f64>,
    #[serde(skip)]
    pub ultima_z: Option<Array2<f64>>,
    #[serde(skip)]
    pub entrada_guardada: Option<Array2<f64>>,
    pub lr: f64,
}

impl CapaDensa {
    pub fn new(entradas: usize, salidas: usize, lr: f64) -> Self {
        let std_dev = (2.0 / entradas as f64).sqrt();
        Self {
            pesos: Array2::random((entradas, salidas), Normal::new(0.0, std_dev).unwrap()),
            sesgos: Array2::zeros((1, salidas)),
            ultima_z: None,
            entrada_guardada: None,
            lr,
        }
    }
}

impl Layer for CapaDensa {
    fn forward(&mut self, entrada: &Array4<f64>) -> Array4<f64> {
        let (b, c, h, w) = entrada.dim();
        let x_2d = entrada.clone().into_shape((b, c * h * w)).unwrap();
        self.entrada_guardada = Some(x_2d.clone());
        let z = x_2d.dot(&self.pesos) + &self.sesgos;
        self.ultima_z = Some(z.clone());
        let activa = z.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let n_salidas = activa.shape()[1];
        activa.into_shape((b, n_salidas, 1, 1)).unwrap()
    }
    fn backward(&mut self, grad_salida: &Array4<f64>) -> Array4<f64> {
        let x = self.entrada_guardada.as_ref().unwrap();
        let z = self.ultima_z.as_ref().unwrap();
        let (b, s, _, _) = grad_salida.dim();
        let g_salida_2d = grad_salida.clone().into_shape((b, s)).unwrap();
        let sig = z.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let d_sig = sig.mapv(|s_val| s_val * (1.0 - s_val));
        let delta = g_salida_2d * d_sig;
        let grad_w = x.t().dot(&delta);
        let grad_b = delta.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_input = delta.dot(&self.pesos.t());
        self.pesos -= &(grad_w * self.lr);
        self.sesgos -= &(grad_b * self.lr);
        let n_entradas = grad_input.shape()[1];
        grad_input.into_shape((b, n_entradas, 1, 1)).unwrap()
    }
    fn get_weights(&self) -> (Option<Array4<f64>>, Option<Array1<f64>>) {
        let (r, c) = self.pesos.dim();
        let w4 = self.pesos.clone().into_shape((1, 1, r, c)).unwrap();
        (Some(w4), Some(self.sesgos.row(0).to_owned()))
    }
    fn set_weights(&mut self, w: Array4<f64>, b: Array1<f64>) {
        let (_, _, r, c) = w.dim();
        self.pesos = w.into_shape((r, c)).unwrap();
        self.sesgos = b.insert_axis(Axis(0));
    }
}

pub struct RedCNN { pub capas: Vec<Box<dyn Layer + Send>> }
impl RedCNN {
    pub fn new() -> Self { Self { capas: Vec::new() } }
    pub fn agregar(&mut self, capa: Box<dyn Layer + Send>) { self.capas.push(capa); }
    pub fn train_step(&mut self, x: &Array4<f64>, y: &Array2<f64>) -> f64 {
        let mut salida = x.clone();
        for capa in &mut self.capas { salida = capa.forward(&salida); }
        let (b, s, _, _) = salida.dim();
        let pred_2d = salida.into_shape((b, s)).unwrap();
        let error = &pred_2d - y;
        let loss = error.mapv(|v| v.powi(2)).sum() / b as f64;
        let mut grad = error.clone().into_shape((b, s, 1, 1)).unwrap();
        for capa in self.capas.iter_mut().rev() { grad = capa.backward(&grad); }
        loss
    }
}

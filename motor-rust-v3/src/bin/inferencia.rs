use motor_v3::{RedModular};
use ndarray::prelude::*;
use std::env;

fn main() {
    // 1. Cargar el cerebro
    let red = RedModular::cargar("modelo_modular_v3.json")
        .expect("❌ No se encontró el modelo. ¡Entrénalo primero!");

    println!("🧠 Modelo cargado con éxito. Capas: {}", red.capas.len());

    // 2. Simular una entrada (esto luego será un dibujo o imagen real)
    let argumentos: Vec<String> = env::args().collect();
    if argumentos.len() < 2 {
        println!("💡 Uso: cargo run --bin inferencia -- [datos_de_entrada]");
    }
    
    // Aquí es donde el usuario podrá interactuar
    println!("🎮 Listo para predicción en tiempo real...");
}

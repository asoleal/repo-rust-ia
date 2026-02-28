use motor_rust_v5::data::generate_sensor_data;
use std::thread;
use std::time::{Duration, Instant};
use std::collections::VecDeque;

fn main() {
    println!("--- 📡 MOTOR V5: MONITORIZACIÓN EN TIEMPO REAL (100Hz) ---");
    let (_, _) = (0, 0); // Placeholder para inicialización
    let mut ventana = VecDeque::with_capacity(128);
    
    println!("{:<15} | {:<15} | {:<10}", "Tiempo (ms)", "Valor Sensor", "IA Status");
    println!("{:-<45}", "");

    for t in 0..200 {
        // Simulación de lectura de sensor (Seno con ruido)
        let base_val = (t as f32 * 0.2).sin();
        let ruido = if t > 100 && t < 120 { 1.5 } else { 0.1 }; // Inyectamos una anomalía
        let val = base_val + ruido;
        
        ventana.push_back(val);
        
        if ventana.len() == 128 {
            let inicio_inferencia = Instant::now();
            
            // Simulación de inferencia ultra-rápida f32
            let status = if ruido > 1.0 { "🔥 FALLA!" } else { "✅ OK" };
            let latencia = inicio_inferencia.elapsed();

            if t % 5 == 0 {
                println!("{:<15} | {:<15.4} | {:<10} ({:?})", t * 10, val, status, latencia);
            }
            ventana.pop_front();
        }
        thread::sleep(Duration::from_millis(10)); // Simula 100Hz
    }
    println!("\n--- Monitoreo finalizado ---");
}

use std::fs;

fn main() {
    let nombre_archivo = "dataset_crudo.csv";

    // Leemos el archivo a la memoria (asumiendo que ya lo creaste en el paso anterior)
    let contenido = fs::read_to_string(nombre_archivo)
        .expect("No se pudo leer el archivo. Ejecuta el paso anterior primero.");

    // Aquí guardaremos las edades válidas y limpias
    let mut edades_limpias: Vec<i32> = Vec::new();

    println!("Iniciando pipeline de extracción de edades...\n");

    // Magia de Iteradores:
    // 1. .lines() lee línea por línea.
    // 2. .skip(1) ignora la primera línea (el encabezado "id,edad,salario").
    for linea in contenido.lines().skip(1) {
        
        // linea.split(',') separa el texto "1,25,4500.50" en pedazos.
        // .nth(1) nos da el SEGUNDO pedazo (índice 0 es id, índice 1 es edad).
        // Como .nth() podría no encontrar nada si la línea está mal formada, 
        // devuelve un Option (Ok/Err disfrazado de Some/None). 
        // Usamos 'if let Some(valor)' para extraerlo de forma segura.
        if let Some(edad_str) = linea.split(',').nth(1) {
            
            // Ahora intentamos convertir ese pedazo de texto a un número entero
            match edad_str.trim().parse::<i32>() {
                Ok(edad_valida) => {
                    edades_limpias.push(edad_valida);
                    println!("✅ Edad extraída y guardada: {}", edad_valida);
                },
                Err(_) => {
                    println!("❌ Dato sucio descartado en la columna edad: '{}'", edad_str);
                }
            }
        }
    }

    println!("\n¡Pipeline finalizado!");
    println!("Vector final listo para analizar: {:?}", edades_limpias);
    
    // Un poco de análisis básico
    let suma: i32 = edades_limpias.iter().sum();
    let promedio = suma as f32 / edades_limpias.len() as f32;
    println!("Promedio de las edades válidas: {:.2}", promedio);
}

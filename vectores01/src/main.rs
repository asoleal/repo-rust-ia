fn main() {
    // 1. Nuestros datos crudos (simulando que vienen de un CSV)
    let columna_cruda = ["25", "  42 ", "veinticinco", "100", "  "];

    // 2. Creamos un Vector VACÍO y MUTABLE para guardar los resultados.
    // Le decimos explícitamente que va a contener enteros: Vec<i32>
    let mut columna_limpia: Vec<i32> = Vec::new();

    println!("Iniciando extracción de datos...\n");

    for texto in columna_cruda {
        // Intentamos limpiar y convertir
        match texto.trim().parse::<i32>() {
            Ok(numero) => {
                // Si la conversión es exitosa, usamos .push() para meterlo al Vector
                columna_limpia.push(numero);
                println!("✅ Guardado: {}", numero);
            },
            Err(_) => {
                // Si hay error (como "veinticinco" o "  "), simplemente lo ignoramos
                println!("❌ Descartado por inválido: '{}'", texto);
            }
        }
    }

    // 3. Imprimimos el Vector resultante.
    // Nota: El {:?} es una sintaxis especial de Rust para imprimir colecciones
    // enteras o Structs en modo "debug" (depuración).
    println!("\n¡Proceso terminado!");
    println!("Columna final lista para ML: {:?}", columna_limpia);
    println!("Total de filas válidas: {}", columna_limpia.len());
}

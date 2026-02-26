// Importamos las herramientas necesarias de la librería estándar
use std::fs::{self, File};
use std::io::Write;

fn main() {
    let nombre_archivo = "dataset_crudo.csv";

    // --- 1. CREAR Y ESCRIBIR EL ARCHIVO ---
    println!("1. Creando archivo '{}' en el disco...", nombre_archivo);
    
    // File::create devuelve un Result. Usamos .expect() para manejar el error rápido.
    let mut archivo = File::create(nombre_archivo)
        .expect("¡Fallo catastrófico al crear el archivo!");
    
    // Escribimos un encabezado y filas con datos. Usamos writeln! (macro para escribir con salto de línea)
    writeln!(archivo, "id,edad,salario").unwrap();
    writeln!(archivo, "1,25,4500.50").unwrap();
    writeln!(archivo, "2,N/A,5000.00").unwrap(); // Dato sucio (N/A)
    writeln!(archivo, "3,30,3200.75").unwrap();
    
    println!("¡Archivo creado con éxito!\n");


    // --- 2. LEER EL ARCHIVO ---
    println!("2. Leyendo el archivo desde el disco duro...");
    
    // fs::read_to_string lee todo el archivo y lo mete en un String en la RAM.
    let contenido = fs::read_to_string(nombre_archivo)
        .expect("No se pudo leer el archivo. ¿Seguro que existe?");

    println!("--- Contenido Crudo ---");
    print!("{}", contenido); // Usamos print! en vez de println! porque el texto ya tiene saltos de línea
    println!("-----------------------\n");


    // --- 3. PROCESAR LÍNEA POR LÍNEA ---
    println!("3. Recorriendo las filas:");
    
    // .lines() es un iterador mágico de Rust que separa el texto cada vez que ve un '\n'
    for (indice, linea) in contenido.lines().enumerate() {
        if indice == 0 {
            println!("Fila {} (Encabezado): {}", indice, linea);
        } else {
            println!("Fila {} (Datos)     : {}", indice, linea);
        }
    }
}

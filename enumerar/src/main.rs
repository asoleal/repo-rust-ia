// 1. Definimos un Enum donde cada variante guarda datos diferentes
enum ValorCelda {
    Entero(i32),
    Flotante(f64),
    Texto(String),
    Vacio, // Representa un dato faltante (Null)
}

fn main() {
    // Creamos diferentes celdas simulando una fila de un dataset
    let edad = ValorCelda::Entero(28);
    let salario = ValorCelda::Flotante(4500.50);
    let nombre = ValorCelda::Texto(String::from("Ana"));
    let telefono = ValorCelda::Vacio;

    println!("Procesando la fila...");
    analizar_dato(&edad);
    analizar_dato(&salario);
    analizar_dato(&nombre);
    analizar_dato(&telefono);
}

// 2. Usamos 'match' para desempacar y actuar según el tipo de dato
fn analizar_dato(dato: &ValorCelda) {
    match dato {
        // Si es Entero, extraemos el valor interno a la variable 'v'
        ValorCelda::Entero(v) => println!("Dato numérico exacto: {}", v),
        
        // Si es Flotante, extraemos a 'f'
        ValorCelda::Flotante(f) => println!("Dato decimal: {}", f),
        
        // Si es Texto, extraemos la referencia al String
        ValorCelda::Texto(t) => println!("Dato categórico (texto): {}", t),
        
        // Si está vacío, no hay nada que extraer
        ValorCelda::Vacio => println!("¡ADVERTENCIA! Dato faltante (NaN/Null)"),
    }
}

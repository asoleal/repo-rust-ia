fn main() {
    // Simulamos una columna leída de un CSV (todo entra como texto inicialmente)
    // Tenemos números limpios, números con espacios y basura.
    let columna_cruda = ["25", "  42 ", "veinticinco", 100];

    println!("Iniciando limpieza de columna...\n");

    for texto in columna_cruda {
        // 1. .trim() quita los espacios en blanco.
        // 2. .parse::<i32>() intenta convertir el texto a un entero de 32 bits.
        // 3. 'intento' NO es un i32. Es un Result<i32, Error>.
        let intento = texto.trim().parse::<i32>();

        // Usamos match para desempaquetar el Result de forma segura
        match intento {
            Ok(numero) => {
                // ¡Éxito! Aquí 'numero' ya es un i32 real que puedes sumar o multiplicar
                println!("✅ Éxito: '{}' se convirtió en el entero {}", texto, numero);
            },
            Err(_) => {
                // El '_' ignora el detalle exacto del error. 
                // Aquí decidimos qué hacer: ignorar, poner 0, o registrar el error.
                println!("❌ Error: '{}' no es un número válido.", texto);
            }
        }
    }
}

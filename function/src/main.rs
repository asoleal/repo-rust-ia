fn main() {
    let dataset = String::from("Dataset de 50GB");

    // Le pasamos un préstamo (referencia) usando '&'
    analizar_datos(&dataset);

    // Como solo lo prestamos, ¡podemos seguir usándolo aquí!
    println!("El main sigue siendo el dueño de: {}", dataset);
}

// La función especifica que espera recibir un préstamo (&String), no el valor entero
// Si quitáramos el '&' aquí y arriba, el 'main' perdería la variable 'dataset'.
fn analizar_datos(datos: &String) {
    println!("Analizando... los datos tienen una longitud de {} bytes.", datos.len());
}

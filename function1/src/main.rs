fn main() {
    // 1. main es el dueño original (lo hacemos mut para poder reasignarlo luego)
    let mut dataset = String::from("Dataset de 50GB");
    println!("1. main tiene: {}", dataset);

    // 2. CEDEMOS la propiedad. Fíjate que NO usamos '&'.
    // Para recuperar los datos, capturamos lo que devuelve la función
    // y sobrescribimos nuestra variable 'dataset'.
    dataset = procesar_y_devolver(dataset);

    // 4. ¡Éxito! main recuperó la propiedad y los datos procesados.
    println!("4. main recuperó: {}", dataset);
}

// La función recibe un 'String' directo, no una referencia. ¡Toma la propiedad!
// El '-> String' indica que esta función va a devolver la propiedad de un String.
fn procesar_y_devolver(mut datos: String) -> String {
    println!("2. La función ahora es dueña exclusiva de: {}", datos);
    
    // Como somos dueños, podemos modificar los datos a nuestro antojo
    datos.push_str(" -> [Limpio y Normalizado]");
    println!("3. La función modificó los datos internamente.");

    // DEVOLVEMOS LA PROPIEDAD
    // En Rust, la última línea de una función SIN punto y coma (;) 
    // se convierte automáticamente en el valor de retorno (return).
    datos
}

fn main() {
    // 1. La variable debe ser mutable desde el principio
    let mut dataset = String::from("Dataset de 50GB");
    println!("1. main original: {}", dataset);

    // 2. Le PRESTAMOS los datos de forma MUTABLE usando '&mut'
    // No perdemos la propiedad, solo damos permiso temporal para modificar.
    limpiar_datos(&mut dataset);

    // 4. main sigue siendo el dueño y ve los cambios inmediatamente
    println!("4. main final: {}", dataset);
}

// 3. La función especifica que recibe un préstamo mutable (&mut String)
// No devuelve nada (no hay '-> String') porque no necesita devolver la propiedad.
fn limpiar_datos(datos: &mut String) {
    println!("2. Función limpiando datos...");
    // Modificamos el texto original directamente en su lugar de memoria
    datos.push_str(" -> [Filtrado y Limpio]");
    println!("3. Función terminó.");
}

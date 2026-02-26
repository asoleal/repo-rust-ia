// 1. Definimos SOLO los datos
struct Dataset {
    nombre: String,
    filas: u32,       // Entero sin signo de 32 bits
    esta_limpio: bool,
}

// 2. Definimos el COMPORTAMIENTO (Métodos)
impl Dataset {
    // Esto es como un "Constructor" o un __init__ en Python.
    // No toma 'self' como parámetro. Es una función asociada al tipo.
    fn nuevo(nombre: &str, filas: u32) -> Dataset {
        Dataset {
            nombre: String::from(nombre),
            filas, // Sintaxis corta si la variable se llama igual que el campo
            esta_limpio: false,
        }
    }

    // Método de solo lectura. Toma prestada la instancia con '&self'
    fn mostrar_info(&self) {
        println!("Dataset '{}': {} filas. ¿Limpio?: {}", 
                 self.nombre, self.filas, self.esta_limpio);
    }

    // Método que modifica los datos. Necesita un préstamo mutable con '&mut self'
    fn limpiar(&mut self) {
        println!("Ejecutando pipeline de limpieza en '{}'...", self.nombre);
        // Filtramos datos malos (simulado)
        self.filas -= 100; 
        self.esta_limpio = true;
    }
}

fn main() {
    // Instanciamos usando nuestro "constructor".
    // IMPORTANTE: Debe ser 'mut' porque vamos a llamar a 'limpiar', 
    // el cual requiere '&mut self'.
    let mut mi_data = Dataset::nuevo("Usuarios_2026", 15000);

    // Llamamos al método de lectura (pasa &self automáticamente)
    mi_data.mostrar_info();

    // Llamamos al método de mutación (pasa &mut self automáticamente)
    mi_data.limpiar();

    // Verificamos los cambios
    mi_data.mostrar_info();
}

fn main() {
    // 1. Tipos de datos: Explícito vs Inferido
    let entero: i32 = 42; // i32 = Entero de 32 bits
    let flotante = 3.1416; // Rust infiere f64 (flotante de 64 bits) por defecto
    
    println!("Entero: {}, Flotante: {}", entero, flotante);

    // 2. Bucles: ¡Igual que en Python!
    // En C++ harías: for(int i=1; i<=3; i++)
    // En Python harías: for numero in range(1, 4):
    println!("Contando del 1 al 3:");
    for numero in 1..3 {
        println!("Iteración: {}", numero);
    }
}

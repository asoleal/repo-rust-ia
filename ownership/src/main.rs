fn main() {
    // 1. Tipos simples en el Stack (Se copian)
    let x = 5;
    let y = x; // x se copia a y
    println!("x = {}, y = {}", x, y); // ¡Ambos funcionan!

    // 2. Tipos complejos en el Heap (Se MUEVEN)
    let s1 = String::from("Datos Masivos");
    let s2 = s1; // ¡OJO! Aquí s1 NO se copia. s1 le transfiere la propiedad a s2.
                 // A partir de esta línea, s1 queda "inválido".

    println!("s2 es el nuevo propietario: {}", s2);
    
    // Instrucción 2: Descomenta la línea de abajo borrando las dos barras (//)
    println!("Intentando usar s1: {}", s1);
}

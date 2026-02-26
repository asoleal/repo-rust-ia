fn main() {
    let s1 = String::from("Datos Masivos de 10GB");
    
    // En lugar de clonar (gastar RAM) o mover (perder s1), 
    // le PRESTAMOS la referencia a s2 usando '&'
    let s2 = &s1; 

    // ¡Ambos funcionan! s2 solo está "mirando" los datos de s1
    println!("s2 está leyendo la data prestada: {}", s2);
    println!("s1 sigue siendo el dueño original: {}", s1);
}

use motor_rust_native::RedNativa;
use ndarray::prelude::*;
use std::fs::File;
use std::io::Read;

fn cargar_modelo(path: &str) -> RedNativa {
    let mut file = File::open(path).expect("No se pudo abrir el modelo");
    let mut json = String::new();
    file.read_to_string(&mut json)
        .expect("Error al leer el archivo");
    serde_json::from_str(&json).expect("Error al deserializar el JSON")
}

fn main() {
    println!("🧠 Cargando cerebro desde el archivo JSON...");
    let red = cargar_modelo("iris_entrenado_nativo.json");

    // Simular un nuevo dato de Iris que la red NO ha visto
    // Ejemplo: Setosa [5.0, 3.4, 1.5, 0.2]
    let nueva_flor = array![[5.0, 3.4, 1.5, 0.2]];

    println!("🌸 Clasificando nueva flor: {:?}", nueva_flor);

    let prediccion = red.predict_native(&nueva_flor);

    println!("\n📊 Probabilidades: {:?}", prediccion);

    // Encontrar el índice con el valor máximo
    let mut max_val = 0.0;
    let mut clase = 0;
    for (i, &val) in prediccion.iter().enumerate() {
        if val > max_val {
            max_val = val;
            clase = i;
        }
    }

    let nombres = ["Setosa", "Versicolor", "Virginica"];
    println!("✨ Resultado: La flor es una Iris **{}**", nombres[clase]);
}

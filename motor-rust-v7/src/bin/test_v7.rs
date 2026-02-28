use motor_rust_v7::data::generate_sensor_data;

fn main() {
    println!("--- 🛡️ V7: VÍNCULO LIBRERÍA-BINARIO EXITOSO ---");
    let (x, _) = generate_sensor_data(1, 128);
    println!("Muestra generada con éxito. Forma: {:?}", x.dim());
}

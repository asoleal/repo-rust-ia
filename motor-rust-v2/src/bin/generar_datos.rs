use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    let mut file = File::create("stress_test.csv")?;
    writeln!(file, "f1,f2,f3,f4,target")?;
    for i in 0..1_000_000 {
        let f1 = (i as f64).sin();
        let f2 = (i as f64).cos();
        let f3 = (i as f64 * 0.1).sin();
        let f4 = (i as f64 * 0.1).cos();
        let target = if f1 + f2 > 0.0 { 0 } else if f3 > 0.5 { 1 } else { 2 };
        writeln!(file, "{:.4},{:.4},{:.4},{:.4},{}", f1, f2, f3, f4, target)?;
    }
    println!("✅ Archivo stress_test.csv de 1,000,000 de filas creado.");
    Ok(())
}

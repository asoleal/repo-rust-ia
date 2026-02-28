use std::fs::File;
use std::io::{Write, BufWriter};

fn main() -> std::io::Result<()> {
    let file = File::create("stress_test.csv")?;
    let mut writer = BufWriter::new(file);
    
    writeln!(writer, "f1,f2,f3,f4,target")?;
    
    println!("Generando 1,000,000 de filas...");
    for i in 0..1_000_000 {
        let f = i as f64 * 0.001;
        let f1 = f.sin();
        let f2 = f.cos();
        let f3 = (f * 0.5).sin();
        let f4 = (f * 0.5).cos();
        
        let target = if f1 + f2 > 0.5 { 0 } else if f3 > 0.0 { 1 } else { 2 };
        
        writeln!(writer, "{:.4},{:.4},{:.4},{:.4},{}", f1, f2, f3, f4, target)?;
    }
    println!("✅ stress_test.csv creado exitosamente.");
    Ok(())
}

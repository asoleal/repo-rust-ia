import regresion_lineal
import polars as pl
import numpy as np
import time

# 1. Generamos 1,000,000 de datos sintéticos usando Numpy
print("Generando 1,000,000 de datos...")
n = 1_000_000
x_data = np.linspace(0, 100, n)
# Creamos una relación y = 3.5x + 12.2 con un poco de ruido aleatorio
y_data = 3.5 * x_data + 12.2 + np.random.normal(0, 1, n)

# 2. Los metemos en un DataFrame de Polars (opcional, pero para que veas el flujo)
df = pl.DataFrame({
    "caracteristica": x_data,
    "objetivo": y_data
})

# 3. Preparamos nuestro modelo de Rust
#modelo = regresion_lineal.RegresionLineal(0.0001, 100) # Menos iteraciones porque hay mucha data
modelo = regresion_lineal.RegresionLineal(0.00001, 500)
print(f"Entrenando con {n} datos en Rust...")
start_time = time.time()

# Convertimos las columnas de Polars a listas de Python (que Rust entiende como Vec<f64>)
modelo.entrenar(df["caracteristica"].to_list(), df["objetivo"].to_list())

end_time = time.time()

print("-" * 30)
print(f"¡Entrenamiento completado en {end_time - start_time:.4f} segundos!")
print(f"Peso (w) encontrado: {modelo.peso:.4f} (Esperado: 3.5)")
print(f"Sesgo (b) encontrado: {modelo.sesgo:.4f} (Esperado: 12.2)")
print("-" * 30)

# 4. Una predicción rápida
print(f"Predicción para x=200: {modelo.predecir([200.0])}")

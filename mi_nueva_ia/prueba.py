import regresion_lineal

# 1. Crear el modelo (Tasa: 0.01, Iteraciones: 5000)
# ¡Nota que usamos la clase que definiste en Rust!
modelo = regresion_lineal.RegresionLineal(0.01, 5000)

# 2. Datos de entrenamiento (y = 2x + 1)
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [3.0, 5.0, 7.0, 9.0, 11.0]

print("Entrenando modelo de Rust desde Python...")
modelo.entrenar(x, y)

# 3. Ver los resultados que Rust guardó
print(f"Peso aprendido (w): {modelo.peso:.4f}")
print(f"Sesgo aprendido (b): {modelo.sesgo:.4f}")

# 4. Predecir valores nuevos
x_nuevos = [6.0, 10.0]
predicciones = modelo.predecir(x_nuevos)

print(f"Para x={x_nuevos}, Rust predice: {predicciones}")

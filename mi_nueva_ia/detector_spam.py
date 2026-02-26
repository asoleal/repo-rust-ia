import regresion_lineal # El nombre del módulo sigue siendo el mismo en Cargo.toml
import numpy as np

# 1. Instanciamos el modelo logístico
# Tasa de aprendizaje un poco más alta para clasificación
modelo = regresion_lineal.RegresionLogistica(0.1, 1000)

# 2. Datos de entrenamiento: Longitud de mensaje -> ¿Es Spam?
# [Longitud: 5, 10, 15] -> No es Spam (0)
# [60, 80, 100] -> Es Spam (1)
x = [5.0, 10.0, 15.0, 60.0, 80.0, 100.0]
y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

print("Entrenando clasificador de Rust...")
modelo.entrenar(x, y)

# 3. Probamos con mensajes nuevos
mensajes_nuevos = [8.0, 75.0] # Uno corto y uno largo
probabilidades = modelo.predecir_probabilidad(mensajes_nuevos)

print("-" * 30)
for m, p in zip(mensajes_nuevos, probabilidades):
    categoria = "SPAM" if p > 0.5 else "NORMAL"
    print(f"Mensaje longitud {m}: Probabilidad de Spam: {p:.4f} -> {categoria}")

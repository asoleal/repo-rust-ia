import regresion_lineal # El nombre del modulo en Cargo.toml sigue igual
import numpy as np

# 1. Instanciamos la IA Multiple
# Parametros: (tasa_aprendizaje, iteraciones, num_caracteristicas)
# Como enviamos [longitud, enlaces, exclamaciones], num_caracteristicas = 3
ia = regresion_lineal.IAMultiple(0.01, 5000, 3)

# 2. Datos de entrenamiento (X es una lista de listas)
# [Longitud, Enlaces, Exclamaciones]
X_train = [
    [5.0, 0.0, 1.0],   # Corto, 0 links, 1 (!) -> Normal
    [15.0, 0.0, 0.0],  # Corto, 0 links, 0 (!) -> Normal
    [60.0, 2.0, 5.0],  # Largo, 2 links, 5 (!) -> Spam
    [80.0, 3.0, 10.0], # Muy largo, 3 links, 10 (!) -> Spam
    [10.0, 1.0, 0.0],  # Corto, 1 link, 0 (!) -> Normal
    [100.0, 5.0, 2.0], # Muy largo, 5 links, 2 (!) -> Spam
]
Y_train = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]

print("Entrenando IA Multivariable en Rust...")
ia.entrenar(X_train, Y_train)

# 3. Verificamos los pesos aprendidos
print("-" * 30)
print(f"Pesos finales: {ia.pesos}")
print(f"Sesgo final: {ia.sesgo}")
print("-" * 30)

# 4. Predicción con un caso nuevo
# Un mensaje de longitud 50, con 2 enlaces y 3 signos de exclamación
caso_nuevo = [[50.0, 2.0, 3.0]]
# Como no definimos 'predecir_probabilidad' en el nuevo lib.rs,
# podemos añadirla o calcularla manualmente con los pesos si quieres ir mas profundo.

# Un mensaje sospechoso: no es muy largo (20), pero tiene 4 enlaces y 8 exclamaciones
sospechoso = [[20.0, 4.0, 8.0]]
probabilidad = ia.predecir(sospechoso)[0]

print(f"Resultado para el mensaje sospechoso:")
print(f"Probabilidad de Spam: {probabilidad:.4f}")
print(f"Veredicto: {'SPAM' if probabilidad > 0.5 else 'NORMAL'}")

import regresion_lineal
import time

# 1. Instanciar (3 características)
ia = regresion_lineal.IAMultiple(0.1, 5000, 3)

# 2. Datos de entrenamiento
# [Horas de estudio, Ejercicios realizados, Horas de sueño] -> [Aprobó (1) / Reprobó (0)]
x_train = [
    [10.0, 20.0, 8.0], [2.0, 5.0, 4.0], [15.0, 30.0, 7.0],
    [1.0, 2.0, 10.0],  [8.0, 15.0, 6.0], [5.0, 10.0, 5.0]
]
y_train = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

# 3. Medir tiempo de entrenamiento matricial
start = time.time()
ia.entrenar(x_train, y_train)
end = time.time()

print(f"🚀 Entrenamiento completado en {end - start:.6f} segundos.")
print(f"Pesos finales: {ia.pesos}")

# 4. Guardar el modelo matricial
ia.guardar("modelo_final.json")

# 5. Predicción
estudiante_nuevo = [[12.0, 25.0, 7.0]]
prob = ia.predecir(estudiante_nuevo)[0]
print(f"\nResultado para nuevo estudiante: {prob:.4f}")
print(f"¿Aprobará?: {'SÍ' if prob > 0.5 else 'NO'}")

import mi_motor_rust as nn
import numpy as np

# 1. Datos de entrenamiento (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# 2. Configuración de la Red
# Arquitectura: 2 entradas -> 4 neuronas ocultas -> 1 salida
# Un learning rate de 0.1 es bueno para empezar
red = nn.RedDinamica([2, 4, 1], 0.1)

print("--- Entrenando Red Neuronal ---")

# 3. Bucle de entrenamiento
epochs = 5000
for epoch in range(epochs):
    red.train_py(X, y)
    
    if epoch % 500 == 0:
        # Calculamos el error cuadrático medio simple para ver progreso
        predicciones = red.forward_py(X)
        loss = np.mean((y - predicciones) ** 2)
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# 4. Verificación final
print("\n--- Resultados Finales ---")
final_preds = red.forward_py(X)
for i in range(len(X)):
    print(f"Entrada: {X[i]} | Predicción: {final_preds[i][0]:.4f} | Esperado: {y[i][0]}")

# 5. Guardar el cerebro entrenado
red.save("modelo_xor_entrenado.json")
print("\n✅ Modelo guardado.")

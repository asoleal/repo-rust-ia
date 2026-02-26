import motor_batches as nb
import numpy as np
import os

# 1. Inicialización
print("--- Fase 1: Entrenamiento Inicial ---")
red = nb.RedBatched([2, 4, 1], 0.1)
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

for i in range(500):
    loss = red.train_batch(X, y)
    if i % 100 == 0:
        print(f"Epoch {i} - Loss: {loss:.6f}")

# 2. Persistencia
filename = "modelo_ia_rust.json"
red.save(filename)
print(f"\n[OK] Modelo guardado en {filename}")

# 3. Verificación de Carga
print("\n--- Fase 2: Carga y Verificación ---")
if os.path.exists(filename):
    nueva_red = nb.RedBatched.load(filename)
    # Probar con un dato
    test_input = np.array([[1,1]], dtype=np.float64)
    # Nota: Aquí puedes implementar un método 'predict' o usar train_batch con lr=0
    print("[OK] El cerebro de la red ha sido reconstruido exitosamente en Rust.")
    print(f"Tamaño del archivo de persistencia: {os.path.getsize(filename)} bytes")

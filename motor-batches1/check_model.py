import motor_batches
import numpy as np
import os

# Crear carpeta si no existe
if not os.path.exists('models'):
    os.makedirs('models')

# Crear una red pequeña y entrenarla mínimamente o solo inicializarla
red = motor_batches.RedBatched([4, 8, 3], 0.1)

# Guardar el modelo
red.save("models/iris_v1.json")

if os.path.exists("models/iris_v1.json"):
    print("✅ ¡Cerebro exportado con éxito a models/iris_v1.json!")
else:
    print("❌ Error al exportar.")

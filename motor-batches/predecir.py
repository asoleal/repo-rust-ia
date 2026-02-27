import motor_batches as nb
import numpy as np

# Cargar el modelo generado en la fase de entrenamiento
try:
    red = nb.RedBatched.load("modelo_ia_rust.json")
    X_test = np.array([[0.0, 1.0]])
    prediccion = red.forward(X_test)
    print(f"\n--- Resultado de la IA en la Nube ---")
    print(f"Entrada [0, 1] -> Predicción: {prediccion[0][0]:.4f}")
except:
    print("Error: Asegúrate de que el modelo_ia_rust.json exista.") 
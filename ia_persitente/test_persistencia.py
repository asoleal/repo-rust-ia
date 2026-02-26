import regresion_lineal
import os

# 1. Crear y Entrenar un modelo
print("--- Fase 1: Entrenamiento ---")
ia = regresion_lineal.IAMultiple(0.1, 1000, 2) # 2 características
x = [[1.0, 2.0], [2.0, 1.0], [5.0, 5.0]]
y = [0.0, 0.0, 1.0]

ia.entrenar(x, y)
print(f"Pesos originales: {ia.pesos}")

# 2. Guardar a disco
nombre_archivo = "modelo_memoria.json"
ia.guardar(nombre_archivo)

# 3. Borrar la instancia de la memoria
del ia
print("\n--- IA original eliminada de la memoria ---")

# 4. Cargar desde el archivo
if os.path.exists(nombre_archivo):
    print("--- Fase 2: Resurrección ---")
    ia_resucitada = regresion_lineal.IAMultiple.cargar(nombre_archivo)
    print(f"Pesos recuperados: {ia_resucitada.pesos}")

    # Probar que sigue funcionando
    test = [[4.5, 4.8]]
    prob = ia_resucitada.predecir(test)
    print(f"Predicción post-resurrección: {prob[0]:.4f}")

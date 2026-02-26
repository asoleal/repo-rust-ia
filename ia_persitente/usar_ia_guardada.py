import regresion_lineal

# 1. Cargamos el "cerebro" desde el archivo JSON
# No necesitamos crear un objeto con 'new', cargar() nos devuelve uno listo
ia_cargada = regresion_lineal.IAMultiple.cargar("modelo_memoria.json")

print("\n--- IA Resucitada ---")
print(f"Pesos que recuerda: {ia_cargada.pesos}")
print(f"Sesgo que recuerda: {ia_cargada.sesgo}")

# 2. Hagamos predicciones nuevas
# El formato debe ser una lista de listas: [[caract1, caract2]]
pruebas = [
    [1.0, 1.0],   # Debería dar algo bajo (cerca de 0)
    [5.0, 5.0],   # Debería dar algo alto (cerca de 1)
    [10.0, 0.0]   # Un caso extremo nuevo
]

print("\n--- Nuevas Predicciones ---")
resultados = ia_cargada.predecir(pruebas)

for i, prob in enumerate(resultados):
    veredicto = "SPAM / POSITIVO" if prob > 0.5 else "NORMAL / NEGATIVO"
    print(f"Entrada {pruebas[i]}: Probabilidad {prob:.4f} -> {veredicto}")

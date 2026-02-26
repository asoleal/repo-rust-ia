import red_neuronal

# 2 entradas, 4 neuronas ocultas
red = red_neuronal.RedNeuronal(0.5, 10000, 2, 4)

# Datos (Estudio, Sueño) -> Aprobado
x = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
y = [1.0, 0.0, 1.0, 0.0]

red.entrenar(x, y)

# Predicción
print(f"Predicción [1, 1]: {red.predecir([[1.0, 1.0]])}")
red.guardar("red_profunda.json")

# Modifica tu test_red.py con estos datos:
x = [[0,0], [0,1], [1,0], [1,1]]
y = [0.0, 1.0, 1.0, 0.0]  # XOR: Solo es 1 si las entradas son diferentes

red.entrenar(x, y)

# Predicción
print(f"Predicción [1, 1]: {red.predecir([[1.0, 1.0]])}")
red.guardar("red_profunda.json")


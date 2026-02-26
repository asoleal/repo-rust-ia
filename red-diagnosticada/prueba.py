import mi_motor_rust as nn
import numpy as np
import matplotlib.pyplot as plt

# Datos XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

red = nn.RedDinamica([2, 8, 1], 0.1)

historico_loss = []
historico_salud = []

print("Entrenando con monitoreo de salud...")
for epoch in range(2000):
    stats = red.train_diagnostico(X, y)
    historico_loss.append(stats.loss)
    historico_salud.append(stats.alive_neurons_ratio)

# Visualización del Rigor Científico
fig, ax1 = plt.subplots()

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(historico_loss, color='tab:red', label='MSE Loss')

ax2 = ax1.twinx()
ax2.set_ylabel('Neuronas Vivas (Ratio)', color='tab:blue')
ax2.plot(historico_salud, color='tab:blue', alpha=0.5, label='Salud ReLU')

plt.title("Análisis de Convergencia y Estabilidad (Rust Engine)")
plt.show()

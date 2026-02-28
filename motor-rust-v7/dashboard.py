import ctypes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# 1. Configurar el puente con la librería de Rust
lib_path = os.path.abspath("target/release/libmotor_rust_v7.so")
motor_lib = ctypes.CDLL(lib_path)

# Configurar predicción
motor_lib.predict_motor_status.argtypes = [ctypes.POINTER(ctypes.c_float)]
motor_lib.predict_motor_status.restype = ctypes.c_int

# Configurar entrenamiento (Fine-tuning)
motor_lib.train_on_sample.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
motor_lib.train_on_sample.restype = ctypes.c_float

# 2. Configuración de la gráfica
fig, ax = plt.subplots(figsize=(10, 5))
x_data = np.arange(128)
line, = ax.plot(x_data, np.zeros(128), color='#2ecc71', lw=2)
status_text = ax.text(0.5, 0.9, 'Esperando datos...', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

# Variable global para guardar la señal actual y que el teclado pueda usarla
current_signal = np.zeros(128, dtype=np.float32)

def generate_signal():
    t = np.linspace(0, 1, 128)
    is_anomaly = np.random.random() > 0.8 
    freq = 25 if is_anomaly else 5
    noise = np.random.normal(0, 0.3, 128)
    return (np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)

def update(frame):
    global current_signal
    current_signal = generate_signal()
    
    # Inferencia en Rust
    data_ptr = current_signal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    prediction = motor_lib.predict_motor_status(data_ptr)
    
    line.set_ydata(current_signal)
    
    if prediction == 1:
        status_text.set_text("⚠️ ESTADO: FALLA | Presiona 'n' si es error")
        status_text.set_color('#e74c3c')
        line.set_color('#e74c3c')
    else:
        status_text.set_text("✅ ESTADO: NORMAL | Presiona 'f' si es error")
        status_text.set_color('#27ae60')
        line.set_color('#2ecc71')
        
    return line, status_text

# 3. Lógica de Fine-Tuning (Interacción Humano-IA)
def on_key(event):
    if event.key in ['f', 'n']:
        label = 1 if event.key == 'f' else 0
        data_ptr = current_signal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Llamada a la función de Rust que actualiza el modelo .json
        new_loss = motor_lib.train_on_sample(data_ptr, label)
        print(f"🔄 [RUST] Fine-tuning aplicado. Nuevo Loss: {new_loss:.6f}")

fig.canvas.mpl_connect('key_press_event', on_key)
ani = FuncAnimation(fig, update, interval=150, blit=True)

print("🚀 Dashboard iniciado.")
print("Apunta a la ventana de la gráfica y presiona 'f' para forzar Falla o 'n' para Normal.")
plt.show()

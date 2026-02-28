import ctypes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# 1. Configurar el puente con la librería de Rust
lib_path = os.path.abspath("target/release/libmotor_rust_v7.so")
if not os.path.exists(lib_path):
    print("❌ Error: No se encuentra la librería .so. Ejecuta 'cargo build --release' primero.")
    exit()

motor_lib = ctypes.CDLL(lib_path)
motor_lib.predict_motor_status.argtypes = [ctypes.POINTER(ctypes.c_float)]
motor_lib.predict_motor_status.restype = ctypes.c_int

# 2. Configuración de la gráfica
fig, ax = plt.subplots(figsize=(10, 5))
x_data = np.arange(128)
line, = ax.plot(x_data, np.zeros(128), color='#2ecc71', lw=2)
status_text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')

ax.set_ylim(-2.5, 2.5)
ax.set_title("📡 Monitoreo de Motor en Tiempo Real (Rust Engine V7)")
ax.grid(True, alpha=0.3)

def generate_signal():
    """Simula una señal que alterna entre normal y falla"""
    t = np.linspace(0, 1, 128)
    # Cambia la probabilidad para simular fallas intermitentes
    is_anomaly = np.random.random() > 0.8 
    freq = 25 if is_anomaly else 5
    noise = np.random.normal(0, 0.3, 128)
    return (np.sin(2 * np.pi * freq * t) + noise).astype(np.float32), is_anomaly

def update(frame):
    # Generar señal
    signal, _ = generate_signal()
    
    # Enviar a Rust para predicción
    data_ptr = signal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    prediction = motor_lib.predict_motor_status(data_ptr)
    
    # Actualizar gráfica
    line.set_ydata(signal)
    
    if prediction == 1:
        status_text.set_text("⚠️ ESTADO: FALLA DETECTADA")
        status_text.set_color('#e74c3c')
        line.set_color('#e74c3c')
    else:
        status_text.set_text("✅ ESTADO: NORMAL")
        status_text.set_color('#27ae60')
        line.set_color('#2ecc71')
        
    return line, status_text

ani = FuncAnimation(fig, update, interval=100, blit=True)
plt.show()

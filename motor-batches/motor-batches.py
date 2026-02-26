import motor_batches as nb
import numpy as np
import time

# Configuración del experimento
input_size = 784  # Simulando una imagen de MNIST (28x28)
hidden_size = 128
output_size = 10
arquitectura = [input_size, hidden_size, output_size]
lr = 0.01

# Generar datos sintéticos masivos para simular carga de Cloud
N_TOTAL = 10000
X_large = np.random.randn(N_TOTAL, input_size).astype(np.float64)
y_large = np.random.randn(N_TOTAL, output_size).astype(np.float64)

red = nb.RedBatched(arquitectura, lr)

def run_benchmark(batch_size):
    # Ajustamos los datos al batch_size actual
    X_batch = X_large[:batch_size]
    y_batch = y_large[:batch_size]
    
    start_time = time.perf_counter()
    # Ejecutamos 100 iteraciones para promediar
    for _ in range(100):
        red.train_batch(X_batch, y_batch)
    end_time = time.perf_counter()
    
    duration = (end_time - start_time) / 100
    samples_per_sec = batch_size / duration
    return duration, samples_per_sec

print(f"{'Batch Size':<12} | {'Tiempo (ms)':<15} | {'Samples/sec':<15}")
print("-" * 48)

for b_size in [1, 16, 32, 64, 128, 256]:
    t, s = run_benchmark(b_size)
    print(f"{b_size:<12} | {t*1000:<15.4f} | {s:<15.2f}")

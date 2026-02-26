import mi_motor_rust as nn
import numpy as np
import os

def test_motor():
    print("--- Iniciando Test de Red Dinámica ---")
    
    # 1. Definimos una arquitectura: 
    # 5 entradas -> 10 neuronas (capa 1) -> 8 neuronas (capa 2) -> 1 salida
    arquitectura = [5, 10, 8, 1]
    lr = 0.05
    
    try:
        # Inicialización de la clase de Rust
        red = nn.RedDinamica(arquitectura, lr)
        print(f"✅ Red creada con {len(arquitectura)-1} capas de pesos.")

        # 2. Prueba de Forward Pass
        # Generamos 3 ejemplos con 5 características cada uno (3x5)
        datos_entrada = np.random.randn(3, 5).astype(np.float64)
        predicciones = red.forward_py(datos_entrada)
        
        print(f"✅ Forward Pass completado.")
        print(f"Forma de la salida: {predicciones.shape} (Esperado: 3x1)")
        print(f"Valores de salida (Sigmoide):\n{predicciones}")

        # 3. Prueba de Persistencia (Serde)
        nombre_archivo = "modelo_v5.json"
        red.save(nombre_archivo)
        
        if os.path.exists(nombre_archivo):
            print(f"✅ Archivo '{nombre_archivo}' generado correctamente.")
            # Opcional: ver el tamaño del archivo
            size = os.path.getsize(nombre_archivo)
            print(f"Tamaño del modelo: {size} bytes.")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")

if __name__ == "__main__":
    test_motor()

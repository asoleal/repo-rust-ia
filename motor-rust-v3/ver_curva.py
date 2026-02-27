import pandas as pd
import matplotlib.pyplot as plt

try:
    # Leer el CSV con la nueva cabecera
    data = pd.read_csv('training_log.csv')
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoca'], data['loss_train'], label='Train Loss (Aprendizaje)', color='blue', linewidth=2)
    plt.plot(data['epoca'], data['loss_test'], label='Test Loss (Generalización)', color='red', linestyle='--')
    
    plt.title('Telemetría del Motor Rust V3: Convergencia de Red Modular')
    plt.xlabel('Épocas')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Añadir un texto con el resultado final
    final_train = data['loss_train'].iloc[-1]
    final_test = data['loss_test'].iloc[-1]
    plt.annotate(f'Final Test: {final_test:.4f}', 
                 xy=(data['epoca'].iloc[-1], final_test), 
                 xytext=(data['epoca'].iloc[-1]-200, final_test+0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.show()
    print("📈 Gráfica de comparación generada. Busca la 'brecha' entre rojo y azul.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Asegúrate de tener instalados pandas y matplotlib: pip install pandas matplotlib")

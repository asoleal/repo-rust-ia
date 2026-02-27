import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import motor_batches

def main():
    # 1. Carga de datos
    iris = load_iris()
    X, y = iris.data.astype(np.float64), iris.target

    # 2. Preprocesamiento
    # OneHot para 3 clases (Setosa, Versicolor, Virginica)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1)).astype(np.float64)
    
    # Escalado para que ReLU no se sature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=0.2, random_state=42
    )

    # 3. Inicializar RedBatched(arquitectura, lr)
    # 4 entradas -> 8 neuronas ocultas -> 3 salidas
    red = motor_batches.RedBatched([4, 8, 3], 0.1)

    # 4. Entrenamiento
    print("🚀 Entrenando motor_batches en Iris...")
    epochs = 500
    batch_size = 10

    for epoch in range(epochs):
        # Shuffle manual simple
        indices = np.random.permutation(len(X_train))
        X_s, y_s = X_train[indices], y_train[indices]
        
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            b_x = X_s[i:i+batch_size]
            b_y = y_s[i:i+batch_size]
            loss = red.train_batch(b_x, b_y)
            epoch_loss += loss
            
        if epoch % 100 == 0:
            print(f"Época {epoch:03d} | Loss: {epoch_loss/(len(X_train)/batch_size):.4f}")

    # 5. Evaluación
    # Rustia devuelve un PyArray2, lo usamos con numpy directamente
    probabilidades = red.predict(X_test)
    predicciones = np.argmax(probabilidades, axis=1)
    reales = np.argmax(y_test, axis=1)

    accuracy = np.mean(predicciones == reales) * 100
    print(f"\n✅ Resultados en Test:")
    print(f"Precisión: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

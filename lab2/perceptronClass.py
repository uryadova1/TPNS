import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, epochs=1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs + 1
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if i == len(self.weights) - 1:
                activation = z
            else:
                activation = self.relu(z)

            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, X, y, output):
        m = X.shape[0]
        error = output - y

        dZ = error
        dW = np.dot(self.activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db

        for i in range(len(self.weights) - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i + 1].T)
            dZ = dA * self.relu_derivative(self.activations[i + 1])
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)


class WineQualityRegressor(NeuralNetwork):
    def __init__(self, input_size, hidden_layers, learning_rate=0.001, epochs=1000):
        layers = [input_size] + hidden_layers + [1]  # Выходной слой с 1 нейроном
        super().__init__(layers, learning_rate, epochs)

    def train(self, X, y):
        # Преобразование y в 2D массив
        y = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        super().train(X, y)

    def predict(self, X):
        return super().predict(X).flatten()


# class MushroomClassifier(NeuralNetwork):
#     def __init__(self, input_size, hidden_layers, learning_rate=0.01, epochs=1000):
#         layers = [input_size] + hidden_layers + [1]  # Выходной слой с 1 нейроном (бинарная классификация)
#         super().__init__(layers, learning_rate, epochs)
#
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
#
#     def forward(self, X):
#         self.activations = [X]
#         self.z_values = []
#
#         for i in range(len(self.weights)):
#             z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
#             self.z_values.append(z)
#
#             if i == len(self.weights) - 1:
#                 # Выходной слой с сигмоидой для классификации
#                 activation = self.sigmoid(z)
#             else:
#                 # Скрытые слои с ReLU
#                 activation = self.relu(z)
#
#             self.activations.append(activation)
#
#         return self.activations[-1]
#
#     def backward(self, X, y, output):
#         m = X.shape[0]
#         y = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
#
#         # Градиенты для выходного слоя
#         dZ = (output - y)  # Для бинарной кросс-энтропии с сигмоидой
#         dW = np.dot(self.activations[-2].T, dZ) / m
#         db = np.sum(dZ, axis=0, keepdims=True) / m
#
#         self.weights[-1] -= self.learning_rate * dW
#         self.biases[-1] -= self.learning_rate * db
#
#         # Обратное распространение для скрытых слоев
#         for i in range(len(self.weights) - 2, -1, -1):
#             dA = np.dot(dZ, self.weights[i + 1].T)
#             dZ = dA * self.relu_derivative(self.activations[i + 1])
#             dW = np.dot(self.activations[i].T, dZ) / m
#             db = np.sum(dZ, axis=0, keepdims=True) / m
#
#             self.weights[i] -= self.learning_rate * dW
#             self.biases[i] -= self.learning_rate * db
#
#     def predict_proba(self, X):
#         return super().predict(X)
#
#     def predict(self, X, threshold=0.5):
#         return (self.predict_proba(X) >= threshold).astype(int).flatten()

class MushroomClassifier(NeuralNetwork):
    def __init__(self, input_size, hidden_layers, learning_rate=0.01, epochs=1000):
        layers = [input_size] + hidden_layers + [1]
        super().__init__(layers, learning_rate, epochs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            activation = self.relu(z)
            # if i == len(self.weights) - 1:
            #     activation = self.sigmoid(z)
            # else:
            #     activation = self.relu(z)

            self.activations.append(activation)

        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y, output):
        m = X.shape[0]
        y = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)

        dZ = output - y
        dW = np.dot(self.activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db

        for i in range(len(self.weights) - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i + 1].T)
            dZ = dA * self.relu_derivative(self.activations[i + 1])
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def train(self, X, y):
        y = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)

        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                loss = self.compute_loss(y, output)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict_proba(self, X):
        return super().predict(X)

    def predict(self, X, threshold=0.6):
        return (self.predict_proba(X) >= threshold).astype(int).flatten()

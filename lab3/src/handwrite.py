import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from lab3.src.GRU import GRU
from lab3.src.LSTM import LSTM
from lab3.src.RNN import RNN
from lab3.src.preprocessing import preprocessing


def train(model, X_train, y_train, epochs, batch_size):
    for epoch in range(epochs):
        loss = 0
        for i in range(0, len(X_train), batch_size):
            x = X_train[i:i + batch_size]
            y = y_train[i:i + batch_size]
            y_pred = model.forward(x)
            loss += np.mean((y_pred - y) ** 2)
            model.backward(x, y)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss / len(X_train):.6f}")


def predict_sequence(model, X, batch_size):
    preds = []
    for i in range(0, len(X), batch_size):
        pred = model.forward(X[i:i + batch_size])
        preds.extend(pred[:, 0])
    return np.array(preds)


X_train, y_train, X_test, y_test = preprocessing()

epochs = 5
batch_size = 64
learning_rate = 0.001

rnn = RNN(input_size=1, hidden_size=50, output_size=1)
gru = GRU(input_size=1, hidden_size=50, output_size=1)
lstm = LSTM(input_size=1, hidden_size=50, output_size=1)

print("Training RNN")
train(rnn, X_train, y_train, epochs, batch_size)
rnn_preds = predict_sequence(rnn, X_test, batch_size)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:200], label='Actual', color='black')
plt.plot(rnn_preds[:200], label='RNN')
plt.title("Model Predictions vs Actual Values")
plt.xlabel("Time Step")
plt.ylabel("Global Active Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Training GRU")
train(gru, X_train, y_train, epochs, batch_size)
gru_preds = predict_sequence(gru, X_test, batch_size)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:200], label='Actual')
plt.plot(gru_preds[:200], label='GRU')
plt.title("Model Predictions vs Actual Values")
plt.xlabel("Time Step")
plt.ylabel("Global Active Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Training LSTM")
train(lstm, X_train, y_train, epochs, batch_size)

lstm_preds = predict_sequence(lstm, X_test, batch_size)

plt.figure(figsize=(12, 6))
plt.plot(y_test[:200], label='Actual')
plt.plot(lstm_preds[:200], label='LSTM')
plt.title("Model Predictions vs Actual Values")
plt.xlabel("Time Step")
plt.ylabel("Global Active Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

rnn_rmse = mean_squared_error(y_test, rnn_preds) ** 0.5
gru_rmse = mean_squared_error(y_test, gru_preds) ** 0.5
lstm_rmse = mean_squared_error(y_test, lstm_preds) ** 0.5

print(f"RNN RMSE: {rnn_rmse:.6f}")
print(f"GRU RMSE: {gru_rmse:.6f}")
print(f"LSTM RMSE: {lstm_rmse:.6f}")

import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация весов
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        batch_size, time_steps, input_size = inputs.shape
        self.last_inputs = inputs
        self.h_states = [np.zeros((batch_size, self.hidden_size))]

        for t in range(time_steps):
            x_t = inputs[:, t, :]  # (batch_size, input_size)
            h_prev = self.h_states[-1]  # (batch_size, hidden_size)

            h_t = np.tanh(
                x_t @ self.Wxh.T + h_prev @ self.Whh.T + self.bh.T
            )  # broadcasting bh

            self.h_states.append(h_t)

        y = self.h_states[-1] @ self.Why.T + self.by.T  # (batch_size, output_size)
        return y

    def backward(self, inputs, targets, learning_rate=0.001):
        batch_size, time_steps, input_size = inputs.shape

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        h_final = self.h_states[-1]  # (batch_size, hidden_size)
        dy = (h_final @ self.Why.T + self.by.T) - targets  # (batch_size, output_size)

        dWhy += dy.T @ h_final
        dby += np.sum(dy, axis=0).reshape(-1, 1)

        dh = dy @ self.Why  # (batch_size, hidden_size)

        for t in reversed(range(time_steps)):
            h = self.h_states[t + 1]
            h_prev = self.h_states[t]
            dtanh = (1 - h ** 2) * dh  # (batch_size, hidden_size)

            dbh += np.sum(dtanh, axis=0).reshape(-1, 1)
            dWxh += dtanh.T @ inputs[:, t, :]
            dWhh += dtanh.T @ h_prev

            dh = dtanh @ self.Whh  # propagate to previous time step

        # Нормализуем по батчу
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -1, 1, out=dparam)  # чтобы избежать градиентного взрыва

        # Градиентный спуск
        self.Wxh -= learning_rate * dWxh / batch_size
        self.Whh -= learning_rate * dWhh / batch_size
        self.Why -= learning_rate * dWhy / batch_size
        self.bh  -= learning_rate * dbh / batch_size
        self.by  -= learning_rate * dby / batch_size

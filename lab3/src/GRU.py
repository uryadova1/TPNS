import numpy as np


class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxz = np.random.randn(hidden_size, input_size) * 0.01
        self.Whz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bz = np.zeros((hidden_size, 1))

        self.Wxr = np.random.randn(hidden_size, input_size) * 0.01
        self.Whr = np.random.randn(hidden_size, hidden_size) * 0.01
        self.br = np.zeros((hidden_size, 1))

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))

        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, inputs):
        batch_size, time_steps, input_size = inputs.shape
        h = np.zeros((batch_size, self.hidden_size))

        self.inputs = inputs
        self.h_states = [h]
        self.z_states = []
        self.r_states = []
        self.h_tilde_states = []

        for t in range(time_steps):
            x = inputs[:, t, :]
            h_prev = self.h_states[-1]

            z = self.sigmoid(x @ self.Wxz.T + h_prev @ self.Whz.T + self.bz.T)
            r = self.sigmoid(x @ self.Wxr.T + h_prev @ self.Whr.T + self.br.T)
            h_tilde = np.tanh(x @ self.Wxh.T + (r * h_prev) @ self.Whh.T + self.bh.T)
            h = (1 - z) * h_prev + z * h_tilde

            self.z_states.append(z)
            self.r_states.append(r)
            self.h_tilde_states.append(h_tilde)
            self.h_states.append(h)

        y = h @ self.Why.T + self.by.T
        return y

    def backward(self, inputs, targets, learning_rate=0.001):
        batch_size, time_steps, input_size = inputs.shape

        dWxz, dWhz, dbz = np.zeros_like(self.Wxz), np.zeros_like(self.Whz), np.zeros_like(self.bz)
        dWxr, dWhr, dbr = np.zeros_like(self.Wxr), np.zeros_like(self.Whr), np.zeros_like(self.br)
        dWxh, dWhh, dbh = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.bh)
        dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)

        h_final = self.h_states[-1]
        y_pred = h_final @ self.Why.T + self.by.T
        dy = y_pred - targets

        dWhy += dy.T @ h_final
        dby += np.sum(dy, axis=0).reshape(-1, 1)
        dh_next = dy @ self.Why

        for t in reversed(range(time_steps)):
            x = inputs[:, t, :]
            h = self.h_states[t + 1]
            h_prev = self.h_states[t]
            z = self.z_states[t]
            r = self.r_states[t]
            h_tilde = self.h_tilde_states[t]

            dh = dh_next
            dh_tilde = dh * z
            dh_prev = dh * (1 - z)
            dz = dh * (h_tilde - h_prev)
            dz_raw = dz * z * (1 - z)

            dWxz += dz_raw.T @ x
            dWhz += dz_raw.T @ h_prev
            dbz += np.sum(dz_raw, axis=0, keepdims=True).T

            dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)
            dWxh += dh_tilde_raw.T @ x
            dWhh += dh_tilde_raw.T @ (r * h_prev)
            dbh += np.sum(dh_tilde_raw, axis=0, keepdims=True).T

            dr = (dh_tilde_raw @ self.Whh) * h_prev
            dr_raw = dr * r * (1 - r)
            dWxr += dr_raw.T @ x
            dWhr += dr_raw.T @ h_prev
            dbr += np.sum(dr_raw, axis=0, keepdims=True).T

            dh_prev = dh_prev + dz_raw @ self.Whz + dr_raw @ self.Whr + (dh_tilde_raw @ self.Whh) * r
            dh_next = dh_prev

        for param, dparam in [(self.Wxz, dWxz), (self.Whz, dWhz), (self.bz, dbz),
                              (self.Wxr, dWxr), (self.Whr, dWhr), (self.br, dbr),
                              (self.Wxh, dWxh), (self.Whh, dWhh), (self.bh, dbh),
                              (self.Why, dWhy), (self.by, dby)]:
            np.clip(dparam, -1, 1, out=dparam)
            param -= learning_rate * dparam / batch_size

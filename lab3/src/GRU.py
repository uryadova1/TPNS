import numpy as np


class GRU:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size)
        self.bz = np.zeros((hidden_size, 1))
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size)
        self.br = np.zeros((hidden_size, 1))
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.Wy = np.random.randn(1, hidden_size)
        self.by = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        for x in inputs:
            x = x.reshape(-1, 1)
            z = np.vstack((h, x))
            z_t = self.sigmoid(self.Wz @ z + self.bz)
            r_t = self.sigmoid(self.Wr @ z + self.br)
            z_r = np.vstack((r_t * h, x))
            h_tilde = np.tanh(self.Wh @ z_r + self.bh)
            h = (1 - z_t) * h + z_t * h_tilde
        y = self.Wy @ h + self.by
        return y.item()

import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.bc = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bo = np.zeros((hidden_size, 1))
        self.Wy = np.random.randn(1, hidden_size)
        self.by = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for x in inputs:
            x = x.reshape(-1, 1)
            z = np.vstack((h, x))
            f = self.sigmoid(self.Wf @ z + self.bf)
            i = self.sigmoid(self.Wi @ z + self.bi)
            c_tilde = np.tanh(self.Wc @ z + self.bc)
            c = f * c + i * c_tilde
            o = self.sigmoid(self.Wo @ z + self.bo)
            h = o * np.tanh(c)
        y = self.Wy @ h + self.by
        return y.item()

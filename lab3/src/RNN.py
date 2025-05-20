import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.Wx = np.random.randn(hidden_size, input_size)
        self.Wh = np.random.randn(hidden_size, hidden_size)
        self.b = np.zeros((hidden_size, 1))
        self.Wy = np.random.randn(1, hidden_size)
        self.by = np.zeros((1, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        for x in inputs:
            x = x.reshape(-1, 1)
            h = np.tanh(self.Wx @ x + self.Wh @ h + self.b)
        y = self.Wy @ h + self.by
        return y.item()

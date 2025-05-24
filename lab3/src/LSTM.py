import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        concat_size = input_size + hidden_size
        self.Wf = np.random.randn(hidden_size, concat_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, concat_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, concat_size) * 0.1
        self.bc = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, concat_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, inputs):
        """
        inputs: (batch_size, time_steps, input_size)
        returns: output predictions (batch_size, output_size)
        """
        batch_size, time_steps, _ = inputs.shape
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))

        self.cache = []

        for t in range(time_steps):
            x = inputs[:, t, :]
            concat = np.concatenate([h, x], axis=1)

            ft = self.sigmoid(concat @ self.Wf.T + self.bf.T)
            it = self.sigmoid(concat @ self.Wi.T + self.bi.T)
            ct_tilde = self.tanh(concat @ self.Wc.T + self.bc.T)
            c = ft * c + it * ct_tilde
            ot = self.sigmoid(concat @ self.Wo.T + self.bo.T)
            h = ot * self.tanh(c)

            self.cache.append((x, h, c, ft, it, ct_tilde, ot, concat))

        y = h @ self.Wy.T + self.by.T
        return y

    def backward(self, inputs, targets, learning_rate=0.001):
        batch_size, time_steps, input_size = inputs.shape

        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        x_last, h_last, _, _, _, _, _, _ = self.cache[-1]
        y_pred = h_last @ self.Wy.T + self.by.T
        dy = y_pred - targets

        dWy += dy.T @ h_last
        dby += np.sum(dy, axis=0, keepdims=True).T
        dh_next = dy @ self.Wy
        dc_next = np.zeros((batch_size, self.hidden_size))

        for t in reversed(range(time_steps)):
            x, h, c, ft, it, ct_tilde, ot, concat = self.cache[t]

            dht = dh_next
            dct = dht * ot * (1 - self.tanh(c) ** 2) + dc_next

            dot = dht * self.tanh(c)
            dot_raw = dot * ot * (1 - ot)

            dWo += dot_raw.T @ concat
            dbo += np.sum(dot_raw, axis=0, keepdims=True).T
            dconcat_o = dot_raw @ self.Wo

            dft = dct * self.cache[t-1][2] if t > 0 else dct * 0
            dft_raw = dft * ft * (1 - ft)

            dWf += dft_raw.T @ concat
            dbf += np.sum(dft_raw, axis=0, keepdims=True).T
            dconcat_f = dft_raw @ self.Wf

            dit = dct * ct_tilde
            dit_raw = dit * it * (1 - it)

            dWi += dit_raw.T @ concat
            dbi += np.sum(dit_raw, axis=0, keepdims=True).T
            dconcat_i = dit_raw @ self.Wi

            dct_tilde = dct * it
            dct_tilde_raw = dct_tilde * (1 - ct_tilde ** 2)

            dWc += dct_tilde_raw.T @ concat
            dbc += np.sum(dct_tilde_raw, axis=0, keepdims=True).T
            dconcat_c = dct_tilde_raw @ self.Wc

            dconcat = dconcat_o + dconcat_f + dconcat_i + dconcat_c
            dh_next = dconcat[:, :self.hidden_size]
            dc_next = dct * ft

        # Обновление весов
        for param, dparam in [(self.Wf, dWf), (self.bf, dbf),
                              (self.Wi, dWi), (self.bi, dbi),
                              (self.Wc, dWc), (self.bc, dbc),
                              (self.Wo, dWo), (self.bo, dbo),
                              (self.Wy, dWy), (self.by, dby)]:
            np.clip(dparam, -1, 1, out=dparam)
            param -= learning_rate * dparam / batch_size

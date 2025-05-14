import numpy as np
import struct
import gzip
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


# class Conv2D:
#     def __init__(self, in_channels, out_channels, kernel_size):
#         self.k = kernel_size
#         self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
#         self.biases = np.zeros((out_channels, 1))
#
#     def forward(self, x):
#         self.input = x
#         batch_size, in_c, h, w = x.shape
#         out_c, _, k, _ = self.weights.shape
#         out_h = h - k + 1
#         out_w = w - k + 1
#         self.output = np.zeros((batch_size, out_c, out_h, out_w))
#
#         for i in range(batch_size):
#             for oc in range(out_c):
#                 for ic in range(in_c):
#                     for y in range(out_h):
#                         for x_ in range(out_w):
#                             self.output[i, oc, y, x_] += np.sum(
#                                 x[i, ic, y:y + k, x_:x_ + k] * self.weights[oc, ic]
#                             )
#                 self.output[i, oc] += self.biases[oc]
#         return self.output
#
#     def backward(self, d_out, lr):
#         batch_size, in_c, h, w = self.input.shape
#         out_c, _, k, _ = self.weights.shape
#         d_input = np.zeros_like(self.input)
#         d_weights = np.zeros_like(self.weights)
#         d_biases = np.zeros_like(self.biases)
#
#         for i in range(batch_size):
#             for oc in range(out_c):
#                 for ic in range(in_c):
#                     for y in range(h - k + 1):
#                         for x_ in range(w - k + 1):
#                             d_weights[oc, ic] += (
#                                     self.input[i, ic, y:y + k, x_:x_ + k] * d_out[i, oc, y, x_]
#                             )
#                             d_input[i, ic, y:y + k, x_:x_ + k] += (
#                                     self.weights[oc, ic] * d_out[i, oc, y, x_]
#                             )
#                 d_biases[oc] += np.sum(d_out[i, oc])
#
#         self.weights -= lr * d_weights
#         self.biases -= lr * d_biases
#         return d_input


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.k = kernel_size
        self.stride = stride
        self.pad = pad
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(out_channels)

    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        self.col = im2col(x, self.k, self.k, self.stride, self.pad)
        self.col_W = self.weights.reshape(self.out_channels, -1).T  # (C*k*k, out_channels)
        out = np.dot(self.col, self.col_W) + self.biases  # (N*out_h*out_w, out_channels)

        out_h = (H + 2 * self.pad - self.k) // self.stride + 1
        out_w = (W + 2 * self.pad - self.k) // self.stride + 1
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        self.output = out  # <-- Добавь это
        return out

    def backward(self, d_out, lr):
        N, C, H, W = self.input.shape
        dout_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        dW = np.dot(self.col.T, dout_reshaped)  # (C*k*k, out_c)
        dW = dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, self.k, self.k)

        db = np.sum(dout_reshaped, axis=0)

        dcol = np.dot(dout_reshaped, self.col_W.T)
        dx = col2im(dcol, self.input.shape, self.k, self.k, self.stride, self.pad)

        # Update
        self.weights -= lr * dW
        self.biases -= lr * db

        return dx


class AvgPool2D:
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        self.input = x
        batch_size, c, h, w = x.shape
        out_h = h // self.size
        out_w = w // self.size
        output = np.zeros((batch_size, c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size]
                output[:, :, i, j] = np.mean(region, axis=(2, 3))
        self.output = output
        return output

    def backward(self, d_out, lr):
        batch_size, c, h, w = self.input.shape
        d_input = np.zeros_like(self.input)
        for i in range(h // self.size):
            for j in range(w // self.size):
                d = d_out[:, :, i, j][:, :, None, None] / (self.size * self.size)
                d_input[:, :, i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size] += d
        return d_input


class FullyConnected:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.biases = np.zeros((1, out_features))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, d_out, lr):
        d_input = np.dot(d_out, self.weights.T)
        d_weights = np.dot(self.input.T, d_out)
        d_biases = np.sum(d_out, axis=0, keepdims=True)

        self.weights -= lr * d_weights
        self.biases -= lr * d_biases
        return d_input


class LeNet5Manual:
    def __init__(self):
        self.conv1 = Conv2D(1, 6, 5)
        self.pool1 = AvgPool2D(2)
        self.conv2 = Conv2D(6, 16, 5)
        self.pool2 = AvgPool2D(2)
        self.fc1 = FullyConnected(16 * 5 * 5, 120)
        self.fc2 = FullyConnected(120, 84)
        self.fc3 = FullyConnected(84, 10)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = tanh(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = tanh(out)
        out = self.pool2.forward(out)
        out = out.reshape(out.shape[0], -1)
        self.fc1_out = tanh(self.fc1.forward(out))
        self.fc2_out = tanh(self.fc2.forward(self.fc1_out))
        out = self.fc3.forward(self.fc2_out)
        return softmax(out)

    def backward(self, d_out, lr):
        d = self.fc3.backward(d_out, lr)
        d = tanh_derivative(self.fc2_out) * d
        d = self.fc2.backward(d, lr)
        d = tanh_derivative(self.fc1_out) * d
        d = self.fc1.backward(d, lr)
        d = d.reshape(-1, 16, 5, 5)
        d = self.pool2.backward(d, lr)
        d = tanh_derivative(self.conv2.output) * d
        d = self.conv2.backward(d, lr)
        d = self.pool1.backward(d, lr)
        d = tanh_derivative(self.conv1.output) * d
        self.conv1.backward(d, lr)


def pad_images(images):
    return np.pad(images, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')


def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy(predictions, labels):
    n = labels.shape[0]
    log_likelihood = -np.log(predictions[np.arange(n), labels] + 1e-9)
    return np.sum(log_likelihood) / n


def cross_entropy_derivative(predictions, labels):
    n = labels.shape[0]
    grad = predictions
    grad[np.arange(n), labels] -= 1
    return grad / n


model = LeNet5Manual()
lr = 0.01

epochs = 6
# padding 2 to 28x28 -> 32x32

train_images = load_images('train-images-idx3-ubyte.gz')

train_labels = load_labels('train-labels-idx1-ubyte.gz')

X_train = train_images[:, None, :, :].astype(np.float32) / 255.
X_train = pad_images(X_train)
y_train = train_labels

batch_size = 32
print(f"packs: {len(X_train) / batch_size}")

losses = list()
accuracies = list()

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    # cnt = 1
    for i in range(0, len(X_train), batch_size):
        # print(f"pack {cnt}")
        x = X_train[i:i + batch_size]
        y = y_train[i:i + batch_size]
        out = model.forward(x)
        loss = cross_entropy(out, y)
        total_loss += loss
        pred = np.argmax(out)
        if pred == y[0]:
            correct += 1
        grad = cross_entropy_derivative(out, y)
        model.backward(grad, lr)
        # cnt += 1

    avg_loss = total_loss / len(X_train)
    acc = correct / len(X_train)
    losses.append(avg_loss)
    accuracies.append(acc)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4%}")

test_images = load_images('t10k-images-idx3-ubyte.gz')
test_labels = load_labels('t10k-labels-idx1-ubyte.gz')

X_test = test_images[:, None, :, :].astype(np.float32) / 255.0
X_test = np.pad(X_test, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
y_test = test_labels

correct = 0
total = len(X_test)

for i in range(total):
    x = X_test[i:i + 1]
    y_true = y_test[i]
    y_pred = np.argmax(model.forward(x))
    if y_pred == y_true:
        correct += 1

accuracy = correct / total
print(f"Test accuracy: {accuracy:.2%}")

y_true = []
y_preds = []

for i in range(len(X_test)):
    x = X_test[i:i + 1]
    out = model.forward(x)
    pred = np.argmax(out)
    y_preds.append(pred)
    y_true.append(y_test[i])

cm = confusion_matrix(y_true, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix LeNet-5")
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(range(1, len(losses) + 1), losses, color=color, label='val Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper right')
ax1.set_title("val Loss")

color = 'tab:blue'
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(range(1, len(accuracies) + 1), accuracies, color=color,
         label='val Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper left')
ax2.set_title("val Accuracy")

plt.tight_layout()
plt.show()

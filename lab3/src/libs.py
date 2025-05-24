import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from preprocessing import preprocessing

X_train, y_train, X_test, y_test = preprocessing()

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)


class RNNModel(nn.Module):
    def __init__(self, cell_type='RNN', hidden_size=64):
        super().__init__()
        if cell_type == 'RNN':
            self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def train_and_evaluate(cell_type):
    model = RNNModel(cell_type=cell_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(4):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[{cell_type}] Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
        actual = y_test.numpy()
        mse = mean_squared_error(actual, preds)
        rmse = mse ** 0.5
        print(f"[{cell_type}] RMSE = {rmse:.4f}")
        return preds, actual, rmse


results = {}
for cell in ['RNN', 'GRU', 'LSTM']:
    preds, actual, rmse = train_and_evaluate(cell)
    results[cell] = (preds, actual, rmse)


plt.figure(figsize=(12, 6))
plt.title("Сравнение предсказаний моделей")

plt.plot(results['RNN'][1][:200], label="Реальные", color='black', linewidth=2)

for model_name, (preds, _, rmse) in results.items():
    plt.plot(preds[:200], label=f"{model_name} (RMSE: {rmse:.4f})")

plt.xlabel("Время")
plt.ylabel("Значение")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Сравнение RMSE:")
for model, (_, _, rmse) in results.items():
    print(f"{model}: {rmse:.4f}")

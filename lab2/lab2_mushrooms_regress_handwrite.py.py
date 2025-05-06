from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from perceptronClass import Regressor

# === Загрузка и подготовка данных ===
column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises?",
                "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
                "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

df = pd.read_csv("../agaricus-lepiota.data", names=column_names)
df['stalk-root'] = df['stalk-root'].replace('?', np.nan)
df = df.dropna()

df['class'] = df['class'].map({'e': 0.0, 'p': 1.0})
y = df['class'].values

X = df.drop('class', axis=1)
X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

corr_matrix = X_encoded.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]
X_encoded = X_encoded.drop(to_drop, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
model = Regressor(input_size=input_size, hidden_layers=[128,  128], learning_rate=0.05, epochs=30)
model.train(X_train.values, y_train)

y_pred = model.predict(X_test.values)

mse = np.mean((y_pred - y_test) ** 2)
print(f"Test MSE: {mse}")

y_classified = (y_pred >= 0.5).astype(int)
accuracy = np.mean(y_classified == y_test)
print(f"Accuracy from regression outputs: {accuracy}")

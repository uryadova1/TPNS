from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# import tensorflow as tf
# print("GPUs:", tf.config.list_physical_devices('GPU'))


column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises?",
                "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
                "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

df = pd.read_csv("../agaricus-lepiota.data", names=column_names)
df['stalk-root'] = df['stalk-root'].replace('?', np.nan)
df = df.dropna()  # убрать пропуски

y = df['class'].map({'e': 0.0, 'p': 1.0}).values

X = df.drop('class', axis=1)
X_encoded = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# with tf.device('/GPU:0'):

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')  # Регрессия: линейный выход
])

model.compile(optimizer=Adam(learning_rate=0.05),
              loss='mean_squared_error',
              metrics=['mae'])

model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.1)

y_pred = model.predict(X_test)

y_pred_classified = (y_pred.flatten() >= 0.5).astype(int)
y_test_classified = y_test.astype(int)

print("Accuracy:", accuracy_score(y_test_classified, y_pred_classified))

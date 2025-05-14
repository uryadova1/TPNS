import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor",
                "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
                "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

df = pd.read_csv("../../agaricus-lepiota.data", names=column_names)

df['stalk-root'] = df['stalk-root'].replace('?', 'missing')
X = df.drop('class', axis=1)
y = df['class']

le_y = LabelEncoder()
y = le_y.fit_transform(y)

binary_cols = []
categorical_cols = []

for col in X.columns:
    if X[col].nunique() == 2:
        binary_cols.append(col)
    else:
        categorical_cols.append(col)

for col in binary_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Один выход с сигмоидой для бинарной классификации

model.compile(optimizer=Adam(learning_rate=0.005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# with tf.device('/GPU:0'):
model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
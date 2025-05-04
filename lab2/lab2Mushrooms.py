from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder


# матрица ошибок

from perceptronClass import MushroomClassifier

# Загрузка данных
column_names = ["class",
                "cap-shape",
                "cap-surface",
                "cap-color",
                "bruises?",
                "odor",
                "gill-attachment",
                "gill-spacing",
                "gill-size",
                "gill-color",
                "stalk-shape",
                "stalk-root",
                "stalk-surface-above-ring",
                "stalk-surface-below-ring",
                "stalk-color-above-ring",
                "stalk-color-below-ring",
                "veil-type",
                "veil-color",
                "ring-number",
                "ring-type",
                "spore-print-color",
                "population",
                "habitat"]
df = pd.read_csv("../agaricus-lepiota.data", names=column_names)
# print(df.nunique())
#
# cleaned_df = df.dropna()
# print(cleaned_df.describe())

# print(df.head())
# print(df.info())
# print(df.isnull().sum())

df['stalk-root'] = df['stalk-root'].replace('?', 0)
X = df.drop('class', axis=1)
y = df['class']

y = LabelEncoder().fit_transform(y)

binary_cols = []  # ["cap-color", "veil-type"]
categorial_cols = []

for col in X.columns:
    unique_vals = X[col].unique()
    if len(unique_vals) == 2:
        binary_cols.append(col)
    else:
        categorial_cols.append(col)

for col in binary_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

X = pd.get_dummies(X, columns=categorial_cols, drop_first=True, dtype=int)

corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.75)]
X = X.drop(to_drop, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


input_size = X_train.shape[1]
# print(f"input size: {input_size}")
model = MushroomClassifier(input_size, [128, 128], learning_rate=0.005, epochs=200)
model.train(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

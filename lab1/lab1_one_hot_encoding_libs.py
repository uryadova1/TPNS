import pandas as pd
import numpy as np
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# names = open("Mushroom/mushroom/agaricus-lepiota.names", "r", encoding="utf-8").read()

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

print(df.head())
print(df.info())
print(df.isnull().sum())
print(f"Дубликаты: {df.duplicated().any()}")

df['stalk-root'] = df['stalk-root'].replace('?', 0)
X = df.drop('class', axis=1)
y = df['class']

y = LabelEncoder().fit_transform(y)

binary_cols = []  # ["cap-color", "veil-type"]
categorial_cols = []  # [col for col in X if col not in binary_cols]

for col in X.columns:
    unique_vals = X[col].unique()
    if len(unique_vals) == 2:
        binary_cols.append(col)
    else:
        categorial_cols.append(col)

print("Бинарные:", binary_cols)
print("Категориальные:", categorial_cols)

for col in binary_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

X = pd.get_dummies(X, columns=categorial_cols, drop_first=True, dtype=int)

print(pd.Series(y).value_counts())

corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
X = X.drop(to_drop, axis=1)
print(f"Удалено {len(to_drop)} коррелирующих признаков: {to_drop}")

plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Матрица корреляций (до удаления)")
plt.show()

clf = DecisionTreeClassifier(criterion='entropy')  # C4.5 использует энтропию
clf.fit(X, y)
importance = clf.feature_importances_
gain_ratio = pd.Series(importance, index=X.columns).sort_values(ascending=False)
print("Топ 10 признаков по Gain Ratio:")
print(gain_ratio.head(10))

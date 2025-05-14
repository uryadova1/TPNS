import pandas as pd
import numpy as np
import seaborn as sns
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def conditional_entropy(X_col, y):
    unique_values = np.unique(X_col)
    cond_entropy = 0.0
    for value in unique_values:
        subset_y = y[X_col == value]
        cond_entropy += (len(subset_y) / len(y)) * entropy(subset_y)
    return cond_entropy


def information_gain(X_col, y):
    return entropy(y) - conditional_entropy(X_col, y)


def split_information(X_col):
    unique_values, counts = np.unique(X_col, return_counts=True)
    probabilities = counts / len(X_col)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def gain_ratio(X_col, y):
    ig = information_gain(X_col, y)
    si = split_information(X_col)
    return ig / (si + 1e-10)


if __name__ == "__main__":
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
    df = pd.read_csv("../../agaricus-lepiota.data", names=column_names)

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

    print("Бинарные:", binary_cols)
    print("Категориальные:", categorial_cols)

    for col in binary_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    X = pd.get_dummies(X, columns=categorial_cols, drop_first=True, dtype=int)

    processed_df = pd.concat([X, pd.Series(y, name='class')], axis=1)

    corr_matrix = X.corr().abs()

    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f")
    plt.title("Матрица корреляций (до удаления)")
    plt.show()

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
    X = X.drop(to_drop, axis=1)
    print(f"Удалено {len(to_drop)} коррелирующих признаков: {to_drop}")

    gr_scores = {}
    for col in X.columns:
        gr_scores[col] = gain_ratio(X[col], y)

    sorted_gr = sorted(gr_scores.items(), key=lambda x: x[1], reverse=True)

    print("Топ-10 признаков по Gain Ratio:")
    for feature, score in sorted_gr[:10]:
        print(f"{feature}: {score:.4f}")

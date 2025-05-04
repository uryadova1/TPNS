from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from perceptronClass import WineQualityRegressor

# Загрузка данных
red_wine = pd.read_csv('../winequality-red.data', sep=';')
white_wine = pd.read_csv('../winequality-white.csv', sep=';')

# Объединение датасетов
wine_data = pd.concat([red_wine, white_wine])

# Разделение на признаки и целевую переменную
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание и обучение модели
input_size = X_train.shape[1]
model = WineQualityRegressor(input_size, [64, 32], learning_rate=0.002, epochs=500)
model.train(X_train, y_train)

# Предсказания на тестовых данных
y_pred = model.predict(X_test)

# Оценка модели

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


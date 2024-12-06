import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor

boston = pd.read_csv('boston_house_prices.csv')
X = boston.drop('MEDV', axis=1)
y = boston['MEDV']

X_scaled = scale(X)

p_values = np.linspace(1, 20, 300)  
best_p = None
best_score = float('inf')

kf = KFold(n_splits=10, shuffle=True, random_state=100)

for p in p_values:
    knn = KNeighborsRegressor(n_neighbors=6, weights='distance', p=p)
    scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    
    mean_score = np.mean(scores)
    
    if mean_score < best_score:
        best_score = mean_score
        best_p = p

print(f"Лучшее значение p: {best_p}")
print(f"Наименьшая среднеквадратичная ошибка: {-best_score}")

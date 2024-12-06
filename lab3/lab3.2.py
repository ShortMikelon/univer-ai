import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("wine.csv")

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

kf = KFold(n_splits=10, shuffle=True, random_state=100)

knn = KNeighborsClassifier()

scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')

print(f"Средняя точность классификации без масштабирования: {scores.mean():.4f}")

k_range = range(1, 101)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = k_range[k_scores.index(max(k_scores))]
best_score = max(k_scores)

print(f"Лучшее значение k: {best_k}, точность: {best_score:.4f}")

X_scaled = scale(X)

k_scores_scaled = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
    k_scores_scaled.append(scores.mean())

best_k_scaled = k_range[k_scores_scaled.index(max(k_scores_scaled))]
best_score_scaled = max(k_scores_scaled)

print(f"Лучшее значение k для масштабированных данных: {best_k_scaled}, точность: {best_score_scaled:.4f}")
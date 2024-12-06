import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# 1. Загрузка данных и масштабирование
digits = datasets.load_digits()
data = scale(digits.data)

# 2. Выводим размерность данных, количество признаков, объектов и уникальных значений
n_samples, n_features = data.shape
unique_targets = np.unique(digits.target)
n_clusters = len(unique_targets)

print(f"Размерность данных: {data.shape}")
print(f"Количество признаков: {n_features}")
print(f"Количество объектов: {n_samples}")
print(f"Количество уникальных значений в целевой переменной: {n_clusters}")

# 3. Создаем модель KMeans с параметром init='k-means++'
kmeans_pp = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)

# 4. Метрики ARI и AMI для KMeans с init='k-means++'
start_time = time.time()
kmeans_pp.fit(data)
time_pp = time.time() - start_time
ari_pp = adjusted_rand_score(digits.target, kmeans_pp.labels_)
ami_pp = adjusted_mutual_info_score(digits.target, kmeans_pp.labels_)


# 5. Модель KMeans с параметром init='random'
kmeans_random = KMeans(init='random', n_clusters=n_clusters, n_init=10)

start_time = time.time()
kmeans_random.fit(data)
time_random = time.time() - start_time
ari_random = adjusted_rand_score(digits.target, kmeans_random.labels_)
ami_random = adjusted_mutual_info_score(digits.target, kmeans_random.labels_)

# 6. Применение PCA
pca = PCA(n_components=n_clusters)
data_pca = pca.fit_transform(data)

# 7. Модель KMeans с параметром init=pca.components_
kmeans_pca = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=10)

start_time = time.time()
kmeans_pca.fit(data)
time_pca = time.time() - start_time
ari_pca = adjusted_rand_score(digits.target, kmeans_pca.labels_)
ami_pca = adjusted_mutual_info_score(digits.target, kmeans_pca.labels_)

# 8. Сравнение всех трёх подходов
print("\nСравнение:")
print(f"KMeans++: ARI = {ari_pp:.4f}, AMI = {ami_pp:.4f}, Время работы: {time_pp:.4f} секунд")
print(f"KMeans Random: ARI = {ari_random:.4f}, AMI = {ami_random:.4f}, Время работы: {time_random:.4f} секунд")
print(f"KMeans PCA: ARI = {ari_pca:.4f}, AMI = {ami_pca:.4f}, Время работы: {time_pca:.4f} секунд")

# 9. Визуализация данных и кластеров на 2D плоскости
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(data)

plt.figure(figsize=(8, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_pp.labels_, cmap='viridis', s=50)

# Отображение центров кластеров
centers_2d = pca_2d.transform(kmeans_pp.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.title("KMeans++ Clustering (PCA reduced to 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

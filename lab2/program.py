import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from sklearn import datasets, decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import scale

# Load the digits dataset
digits = datasets.load_digits()
data = digits.data
target = digits.target
unique_targets = np.unique(target)

# Scale the data
data = scale(data)

print("Размерность данных: ", data.shape)
print(f"Количество признаков: {len(data[0])}")
print(f'Количество объектов: {len(data)}')

# KMeans with k-means++ initialization
kmeans = KMeans(n_clusters=len(unique_targets), n_init=10)
start_time = time.time()
kmeans.fit(data)
kmeans_time = time.time() - start_time

labels = kmeans.labels_
ari_k_means = adjusted_rand_score(target, labels)
ami_k_means = adjusted_mutual_info_score(target, labels)

print(f'Время работы для k_means++: {kmeans_time:.4f} секунд')
print(f'ARI для k_means++: {ari_k_means:.4f}')
print(f'AMI для k_means++: {ami_k_means:.4f}')

# KMeans with random initialization
kmeans_random = KMeans(init="random", n_clusters=len(unique_targets), n_init=10)
start_time = time.time()
kmeans_random.fit(data)
kmeans_random_time = time.time() - start_time

labels_random = kmeans_random.labels_
ari_random = adjusted_rand_score(target, labels_random)
ami_random = adjusted_mutual_info_score(target, labels_random)

print(f'Время работы для random: {kmeans_random_time:.4f} секунд')
print(f'ARI для random: {ari_random:.4f}')
print(f'AMI для random: {ami_random:.4f}')

# PCA for dimensionality reduction
pca = decomposition.PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# KMeans with PCA initialization
kmeans_pca = KMeans(init='k-means++', n_clusters=len(unique_targets), n_init=10)
start_time = time.time()
kmeans_pca.fit(data_reduced)
kmeans_pca_time = time.time() - start_time

labels_pca = kmeans_pca.labels_
ari_pca = adjusted_rand_score(target, labels_pca)
ami_pca = adjusted_mutual_info_score(target, labels_pca)

print(f'Время работы для PCA: {kmeans_pca_time:.4f} секунд')
print(f'ARI для PCA: {ari_pca:.4f}')
print(f'AMI для PCA: {ami_pca:.4f}')

# Plotting the PCA reduced data
plt.figure(figsize=(12, 10))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=target, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.title('PCA Reduced Data with True Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise
from scipy.spatial.distance import pdist, squareform

df_rates = pd.read_csv('datasets/user_ratedmovies.dat', sep='\t')
df_movies = pd.read_csv("datasets/movies.dat", sep='\t', encoding='iso-8859-1')

enc_user = LabelEncoder()
enc_mov = LabelEncoder()

enc_user = enc_user.fit(df_rates.userID.values) 
enc_mov = enc_mov.fit(df_rates.movieID.values)

idx = df_movies.loc[:, 'id'].isin(df_rates.movieID)
df_movies = df_movies.loc[idx]

df_rates.loc[:, 'userID'] = enc_user.transform(df_rates.loc[:, 'userID'].values)
df_rates.loc[:, 'movieID'] = enc_mov.transform(df_rates.loc[:, 'movieID'].values)
df_movies.loc[:, 'id'] = enc_mov.transform(df_movies.loc[:, 'id'].values)

R = coo_matrix((df_rates.rating.values, (df_rates.userID.values, df_rates.movieID.values)))
print("Размер матрицы R:", R.shape)

cosine_similarity = pairwise.cosine_similarity(R.toarray())
print("Размерность косинусной матрицы:", cosine_similarity.shape)

def adjusted_cosine_similarity(u, v):
    intersection = (u > 0) & (v > 0)
    if intersection.sum() == 0: 
        return 0
    else:
        similarity = pairwise.cosine_similarity([u], [v])[0][0]
        return 1 + (-similarity)

distances = pdist(R.toarray(), metric=adjusted_cosine_similarity)
print("Размерность вектора расстояний:", distances.shape)

distance_matrix = squareform(distances)
print("Размерность полученной матрицы расстояний:", distance_matrix.shape)

user_index = 0
user_distances = distance_matrix[user_index]

closest_user_indices = np.argsort(user_distances)[1:11]

closest_users = [(index + 1, user_distances[index]) for index in closest_user_indices]  # +1 для реальных идентификаторов пользователей
print("Ближайшие пользователи к пользователю №1:")
for user_id, distance in closest_users:
    print(f"Пользователь ID: {user_id}, Расстояние: {distance:.4f}")
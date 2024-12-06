import pandas as pd

df_rates=pd.read_csv('datasets/user_ratedmovies.dat',sep='\t')
df_movies=pd.read_csv("datasets/movies.dat",sep='\t',encoding='iso-8859-1')

from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

enc_user=LabelEncoder()
enc_mov=LabelEncoder()

enc_user=enc_user.fit(df_rates.userID.values)
enc_mov=enc_mov.fit(df_rates.movieID.values)

idx=df_movies.loc[:,'id'].isin(df_rates.movieID)
df_movies=df_movies.loc[idx]

df_rates.loc[:,'userID'] = enc_user.transform(df_rates.loc[:,'userID'].values)
df_rates.loc[:,'movieID'] = enc_mov.transform(df_rates.loc[:,'movieID'].values)
df_movies.loc[:, 'id'] = enc_mov.transform(df_movies.loc[:,'id'].values)

from scipy.sparse import coo_matrix
R = coo_matrix((df_rates.rating.values, (df_rates.userID.values,df_rates.movieID.values)))

from scipy.sparse.linalg import svds
u,s,vt = svds(R,k=6)
from sklearn.neighbors import NearestNeighbors

nn=NearestNeighbors(n_neighbors=10)

v=vt.T

nn.fit(v)

_, ind = nn.kneighbors(v, n_neighbors=10)

movie_titles=df_movies.sort_values('id').loc[:,'title'].values
cols=['movie']+['nn_{}'.format(i) for i in range(1,10)]
df_ind_nn=pd.DataFrame(data=movie_titles[ind],columns=cols)

idx=df_ind_nn.movie.str.contains('Terminator 2')

print(df_ind_nn.loc[idx].head(10))
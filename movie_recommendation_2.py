import kagglehub
import pandas as pd
#path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")

#print("Path to dataset files:", path)
file_paths = {
    'u_data': 'movielens-100k-dataset/versions/1/ml-100k/u.data',
    'u_info': 'movielens-100k-dataset/versions/1/ml-100k/u.info',
    'u_item': 'movielens-100k-dataset/versions/1/ml-100k/u.item',
    'u_genre': 'movielens-100k-dataset/versions/1/ml-100k/u.genre',
    'u_user': 'movielens-100k-dataset/versions/1/ml-100k/u.user',
    'u_occupation': 'movielens-100k-dataset/versions/1/ml-100k/u.occupation',
}
df = pd.read_csv(file_paths['u_data'], sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
df = df.drop(['timestamp'], axis=1)
print(df.head())
with open(file_paths['u_info'], 'r') as f:
    u_info = f.read().splitlines()
print("\nu_info List:")
print(u_info)
# 3. Load u.item

item_df = pd.read_csv(file_paths['u_item'], sep='|', encoding='latin-1', header=None,
                     names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                            'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                            'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
item_df = item_df.drop(['video_release_date', 'IMDb_URL'], axis=1)
print(item_df.head())
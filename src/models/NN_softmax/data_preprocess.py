import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict

class MovieDataProcessor:
    def __init__(self, ratings_path: str, movies_path: str):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.user_enc = LabelEncoder()
        self.movie_enc = LabelEncoder()
        
    def load_and_process_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load and process movie and ratings data"""
        # Load ratings
        ratings_cols = ['user id', 'movie id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(
            self.ratings_path, 
            sep='\t', 
            header=None, 
            names=ratings_cols
        )

        # Load movies
        movies_cols = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
        movies_cols = movies_cols.split(' | ')
        movies_df = pd.read_csv(
            self.movies_path,
            sep='|',
            header=None,
            names=movies_cols,
            encoding='latin-1'
        )
        movie_info = movies_df[['movie id', 'movie title']]

        # Merge datasets
        merged_df = pd.merge(ratings_df, movie_info, how='inner', on='movie id')
        refined_df = merged_df.groupby(
            by=['user id', 'movie title'], 
            as_index=False
        ).agg({"rating": "mean"})

        # Encode users and movies
        refined_df['user'] = self.user_enc.fit_transform(refined_df['user id'].values)
        refined_df['movie'] = self.movie_enc.fit_transform(refined_df['movie title'].values)

        # Get metadata
        metadata = {
            'n_users': refined_df['user'].nunique(),
            'n_movies': refined_df['movie'].nunique(),
            'min_rating': refined_df['rating'].min(),
            'max_rating': refined_df['rating'].max()
        }

        return refined_df, metadata

    def prepare_training_data(self, refined_df: pd.DataFrame, test_size: float = 0.1
                            ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Prepare training and validation data"""
        # Extract features and target
        X = refined_df[['user', 'movie']].values
        y = refined_df['rating'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Normalize ratings
        min_rating = refined_df['rating'].min()
        max_rating = refined_df['rating'].max()
        y_train = (y_train - min_rating)/(max_rating - min_rating)
        y_test = (y_test - min_rating)/(max_rating - min_rating)

        # Prepare input arrays
        X_train_array = [X_train[:, 0], X_train[:, 1]]
        X_test_array = [X_test[:, 0], X_test[:, 1]]

        return (X_train_array, y_train), (X_test_array, y_test)

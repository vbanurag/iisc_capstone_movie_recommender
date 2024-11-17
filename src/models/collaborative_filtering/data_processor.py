import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.refined_dataset = None
        self.user_to_movie_df = None
        self.movie_to_user_df = None
        self.movies_list = None
        self.case_insensitive_movies_list = None
        self.movie_dict = None

    def load_data(self):
        """Load and process the initial dataset."""
        # Load ratings data
        column_names = ["user id", "movie id", "rating", "timestamp"]
        self.dataset = pd.read_csv(f'{self.data_path}/u.data', sep='\t', 
                                 header=None, names=column_names)
        
        # Load movies data
        movies_columns = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
        movies_columns = movies_columns.split(' | ')
        self.items_dataset = pd.read_csv(f'{self.data_path}/u.item', sep='|', 
                                       header=None, names=movies_columns, 
                                       encoding='latin-1')

    def process_data(self):
        """Process the loaded data into required format."""
        # Create movie dataset
        movie_dataset = self.items_dataset[['movie id', 'movie title']]
        
        # Merge datasets
        merged_dataset = pd.merge(self.dataset, movie_dataset, 
                                how='inner', on='movie id')
        
        # Create refined dataset
        self.refined_dataset = merged_dataset.groupby(
            by=['user id', 'movie title'], 
            as_index=False
        ).agg({"rating": "mean"})

        # Create user-movie matrix
        self.user_to_movie_df = self.refined_dataset.pivot(
            index='user id',
            columns='movie title',
            values='rating'
        ).fillna(0)

        # Create movie-user matrix
        self.movie_to_user_df = self.refined_dataset.pivot(
            index='movie title',
            columns='user id',
            values='rating'
        ).fillna(0)

        # Create movies list and dictionary
        self.movies_list = list(self.movie_to_user_df.index)
        self.case_insensitive_movies_list = [i.lower() for i in self.movies_list]
        self.movie_dict = {movie: index for index, movie in enumerate(self.movies_list)}
    
    def get_dataf(self):
        return self.user_to_movie_df, self.movie_to_user_df, self.refined_dataset

    def get_sparse_matrices(self):
        """Convert matrices to sparse format."""
        user_to_movie_sparse = csr_matrix(self.user_to_movie_df.values)
        movie_to_user_sparse = csr_matrix(self.movie_to_user_df.values)
        return user_to_movie_sparse, movie_to_user_sparse

    def get_movie_info(self):
        """Return movie related information."""
        return {
            'movies_list': self.movies_list,
            'case_insensitive_movies_list': self.case_insensitive_movies_list,
            'movie_dict': self.movie_dict
        }
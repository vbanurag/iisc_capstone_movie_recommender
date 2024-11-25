import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SVDMovieRecommendation:
    def __init__(self, data_path):
        self.data_path = data_path
        self._load_data()
        self._prepare_utility_matrix()
        self._compute_svd()
        
    def _load_data(self):
        column_names1 = ['user id', 'movie id', 'rating', 'timestamp']
        self.dataset = pd.read_csv(
            os.path.join(self.data_path, 'u.data'), sep='\t', header=None, names=column_names1
        )
        
        d = ('movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure '
             '| Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical '
             '| Mystery | Romance | Sci-Fi | Thriller | War | Western')
        column_names2 = d.split(' | ')
        items_dataset = pd.read_csv(
            os.path.join(self.data_path, 'u.item'), sep='|', header=None, names=column_names2, encoding='latin-1'
        )
        movie_dataset = items_dataset[['movie id', 'movie title']]
        
        # Merge datasets and group by user and movie
        merged_dataset = pd.merge(self.dataset, movie_dataset, how='inner', on='movie id')
        self.refined_dataset = merged_dataset.groupby(
            by=['user id', 'movie title'], as_index=False
        ).agg({"rating": "mean"})
        
        self.unique_users = self.refined_dataset['user id'].unique()
        self.unique_movies = self.refined_dataset['movie title'].unique()
        
        # Create mapping for movies
        self.movies_dict = {movie: i for i, movie in enumerate(self.unique_movies)}
        self.case_insensitive_movies_list = [movie.lower() for movie in self.unique_movies]
    
    def _prepare_utility_matrix(self):
        # Create the utility matrix
        num_users = len(self.unique_users)
        num_movies = len(self.unique_movies)
        self.utility_matrix = np.full((num_movies, num_users), np.nan)
        
        for _, row in self.refined_dataset.iterrows():
            movie_index = self.movies_dict[row['movie title']]
            user_index = row['user id'] - 1
            self.utility_matrix[movie_index, user_index] = row['rating']
        
        # Fill missing values with average ratings
        mask = np.isnan(self.utility_matrix)
        masked_arr = np.ma.masked_array(self.utility_matrix, mask)
        rating_means = np.mean(masked_arr, axis=1).filled(0)
        self.filled_matrix = masked_arr.filled(rating_means[:, np.newaxis])
        self.rating_means = rating_means
    
    def _compute_svd(self):
        # Normalize and compute SVD
        normalized_matrix = self.filled_matrix - self.rating_means[:, np.newaxis]
        self.U, self.S, self.Vt = np.linalg.svd(normalized_matrix.T / np.sqrt(len(self.movies_dict) - 1))
    
    def _top_cosine_similarity(self, data, movie_id, top_n=10):
        movie_row = data[movie_id, :]
        magnitude = np.sqrt(np.einsum('ij,ij->i', data, data))
        magnitude[magnitude == 0] = 1e-10 
        similarity = np.dot(movie_row, data.T) / (magnitude[movie_id] * magnitude)
        return np.argsort(-similarity)[:top_n]
    
    def get_similar_movies(self, movie_name, top_n=10, k=50):
        if movie_name not in self.movies_dict:
            raise ValueError("Movie not found in the dataset.")
        
        sliced = self.Vt.T[:, :k]  # Use the first k components
        movie_id = self.movies_dict[movie_name]
        indexes = self._top_cosine_similarity(sliced, movie_id, top_n)
        
        print(f"\nTop {top_n} movies similar to '{movie_name}':\n")
        for i in indexes[1:]:  # Exclude the movie itself
            print(self.unique_movies[i])
    
    def get_possible_movies(self, movie):
        temp = ''
        possible_movies = self.case_insensitive_movies_list.copy()
        for char in movie.lower():
            out = []
            temp += char
            for name in possible_movies:
                if temp in name:
                    out.append(name)
            if not out:
                return possible_movies
            possible_movies = out
        return possible_movies
    
    def recommender(self):
        try:
            movie_name = input("Enter the movie name: ").strip()
            movie_name_lower = movie_name.lower()
            
            if movie_name_lower not in self.case_insensitive_movies_list:
                raise ValueError
            
            num_recom = int(input("Enter the number of movie recommendations needed: "))
            self.get_similar_movies(self.unique_movies[self.case_insensitive_movies_list.index(movie_name_lower)], num_recom)
        
        except ValueError:
            possible_movies = self.get_possible_movies(movie_name_lower)
            
            if len(possible_movies) == len(self.unique_movies):
                print("Movie name does not exist in the dataset.")
            else:
                print("Did you mean one of these movies?\n")
                print([self.unique_movies[self.case_insensitive_movies_list.index(m)] for m in possible_movies])
                print()
                self.recommender()
    
     def test_model(self, test_size=0.2):
        # Split dataset into train and test
        test_data = self.refined_dataset.sample(frac=test_size, random_state=42)
        train_data = self.refined_dataset.drop(test_data.index)

        # Create utility matrix for the training set
        train_utility_matrix = np.full((len(self.unique_movies), len(self.unique_users)), np.nan)
        for _, row in train_data.iterrows():
            movie_index = self.movies_dict[row['movie title']]
            user_index = row['user id'] - 1
            train_utility_matrix[movie_index, user_index] = row['rating']

        # Fill missing values with average ratings
        train_mask = np.isnan(train_utility_matrix)
        train_masked_arr = np.ma.masked_array(train_utility_matrix, train_mask)
        train_rating_means = np.mean(train_masked_arr, axis=1).filled(0)
        filled_train_matrix = train_masked_arr.filled(train_rating_means[:, np.newaxis])

        # Perform SVD
        normalized_train_matrix = filled_train_matrix - train_rating_means[:, np.newaxis]
        U, S, Vt = np.linalg.svd(normalized_train_matrix.T / np.sqrt(len(self.movies_dict) - 1))

        # Predict ratings on the test set
        predictions = []
        ground_truth = []
        for _, row in test_data.iterrows():
            user_id = row['user id'] - 1
            movie_title = row['movie title']
            if movie_title in self.movies_dict:
                movie_id = self.movies_dict[movie_title]
                predicted_rating = train_rating_means[movie_id] + np.dot(
                    U[user_id, :50], S[:50] * Vt[:50, movie_id]
                )
                predictions.append(predicted_rating)
                ground_truth.append(row['rating'])

        # Calculate RMSE
        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        mae = mean_absolute_error(ground_truth, predictions)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return rmse, mae


if __name__ == "__main__":
    # Create an instance of the recommendation system
    movie_recommender = SVDMovieRecommendation(data_path="data/ml-100k")
    
    # Train or initialize the recommender
    movie_recommender.recommender()
    
    # Test the model and get evaluation metrics
    rmse, mae = movie_recommender.test_model(test_size=0.2)
    
    # Print the results
    print(f"RMSE on the test dataset: {rmse:.4f}")
    print(f"MAE on the test dataset: {mae:.4f}")


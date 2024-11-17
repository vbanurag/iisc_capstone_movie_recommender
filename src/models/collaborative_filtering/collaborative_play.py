from .collab_filtering import MovieRecommender
from .data_processor import DataProcessor
from .helper import MovieFinder
import os

class CollaborativeMovieSuggestion:
    def __init__(self) -> None:
        absolute_path = os.path.abspath('data/ml-100k')
        self.data_processor = DataProcessor(absolute_path)
        print("Loading data...")
        self.data_processor.load_data()
        self.data_processor.process_data()
          
    def _intialize_pre_deps(self):
        # Get sparse matrices and movie info
        user_movie_sparse, movie_user_sparse = self.data_processor.get_sparse_matrices()
        movie_info = self.data_processor.get_movie_info()
        return user_movie_sparse, movie_user_sparse, movie_info
    
    def _intialize_movie_recommendation_engine(self):
        # Initialize recommender and movie finder
        user_to_movie_df, movie_to_user_df, refined_dataset = self.data_processor.get_dataf()
        user_movie_sparse, movie_user_sparse, movie_info = self._intialize_pre_deps() 
        
        recommender = MovieRecommender(
            refined_dataset,
            user_movie_sparse,
            movie_user_sparse,
            user_to_movie_df,
            movie_to_user_df,
        )
        
        movie_finder = MovieFinder(
            movie_info['movies_list'],
            movie_info['case_insensitive_movies_list']
        )
        return recommender, movie_finder
    
    def recommended_movies(self, user_id, n=10):
        recommender, movie_finder = self._intialize_movie_recommendation_engine()

        similar_movies = recommender.get_movie_recommendations(user_id, n)
                    
        print(f"\nTop {n} movies similar to '{user_id}':")
        for i, movie_data in enumerate(similar_movies, 1):
            print(f"{i}. {movie_data}")

    def recommend_movies_user(self, movie, n=10):
       recommender, movie_finder = self._intialize_movie_recommendation_engine()
       
       similar_movies = recommender.get_similar_movies(movie, n)
       print(f"\nTop {n} users similar to Movie {movie}: ")
       for i, movie_data in enumerate(similar_movies, 1):
           print(f"{i}. Recommended Movies:  {movie_data} ")

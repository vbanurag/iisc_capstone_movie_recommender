import os

from .movie_recommender import MovieRecommenderModel

from .data_preprocess import MovieDataProcessor

class MovieRecommendDD:
    def __init__(self):
        
        pass

    def load_data(self):
        rating_path = os.path.abspath('data/ml-100k/u.data')
        movie_path = os.path.abspath('data/ml-100k/u.item')
        self.DataHandler = MovieDataProcessor(rating_path, movie_path)

    def process_data(self):
        refind_df, metadata = self.DataHandler.load_and_process_data()
        self.refined_df = refind_df
        self.metadata = metadata

    def _prep_training_data(self):
        X,Y = self.DataHandler.prepare_training_data(self.refined_df)
        return X,Y
    
    def prepare_model(self):
        X,Y = self._prep_training_data()
        self.movie_model = MovieRecommenderModel(self.metadata['n_users'], self.metadata['n_movies'], self.refined_df)
        self.movie_model.train(X[0],X[1],Y[0],Y[1])


    def get_movie_recommendations(self, user_id):
        self.prepare_model()
        movies_list = self.movie_model.recommender_movie_system(user_id)
        print(f"\nTop  movies for '{user_id}':")
        for i, movie_data in enumerate(movies_list, 1):
            print(f"{i}. {movie_data}") 

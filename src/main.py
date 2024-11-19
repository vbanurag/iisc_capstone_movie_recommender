from models.collaborative_filtering.collaborative_play import CollaborativeMovieSuggestion
from models.NN_softmax.dd_play import MovieRecommendDD


def main():
    # collabMovie = CollaborativeMovieSuggestion()
    # collabMovie.recommended_movies(78)
    # collabMovie.recommend_movies_user('Absolute Power (1997)')

    softmax_dd = MovieRecommendDD()
    softmax_dd.load_data()
    softmax_dd.process_data()
    softmax_dd.prepare_model()
    softmax_dd.get_movie_recommendations(45)
    

if __name__ == "__main__":
    main()
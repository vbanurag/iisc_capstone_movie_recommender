from models.collaborative_filtering.collaborative_play import CollaborativeMovieSuggestion


def main():
    collabMovie = CollaborativeMovieSuggestion()
    collabMovie.recommended_movies(78)
    collabMovie.recommend_movies_user('Absolute Power (1997)')
    

if __name__ == "__main__":
    main()
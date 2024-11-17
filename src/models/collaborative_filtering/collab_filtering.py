from pprint import pprint
from sklearn.neighbors import NearestNeighbors
import numpy as np

class MovieRecommender:
    def __init__(self,refined_dataset, user_to_movie_sparse, movie_to_user_sparse, user_to_movie_df, movie_to_user_df ):
        """
        Initialize the MovieRecommenderSystem with required data and models.
        
        Parameters:
        -----------
        refined_dataset : pandas.DataFrame
            Dataset containing user_id and movie_title columns
        user_to_movie_df : pandas.DataFrame
            User-movie matrix with users as rows and movies as columns
        knn_model : sklearn.neighbors.NearestNeighbors
            Trained KNN model for finding similar users
        """
        self.refined_dataset = refined_dataset
        self.user_to_movie_df = user_to_movie_df
        self.movie_to_user_df = movie_to_user_df
        self.user_to_movie_sparse = user_to_movie_sparse
        self.movie_to_user_sparse = movie_to_user_sparse
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_model_m = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn_movie_model = self.knn_model_m.fit(self.movie_to_user_sparse)
        self.knn_user_model = self.knn_model.fit(self.user_to_movie_sparse)
        self.movies_list = user_to_movie_df.columns

    def get_user_movies(self, user_id):
        """Get movies seen by a specific user."""
        user_movies = list(self.refined_dataset[self.refined_dataset['user id'] == user_id]['movie title'])
        print("Movies seen by the User:")
        pprint(user_movies)
        print("")
        return user_movies

    def get_similar_users(self, user_id, n=5):
        """
        Find similar users using KNN model.
        
        Parameters:
        -----------
        user_id : int
            User ID to find similar users for
        n : int
            Number of similar users to return
            
        Returns:
        --------
        tuple : (similar_users, distances)
        """
        knn_input = np.asarray([self.user_to_movie_df.values[user_id-1]])
        distances, indices = self.knn_user_model.kneighbors(knn_input, n_neighbors=n+1)
        
        print(f"Top {n} users who are very much similar to the User-{user_id} are:")
        print("")
        for i in range(1, len(distances[0])):
            print(f"{i}. User: {indices[0][i]+1} separated by distance of {distances[0][i]}")
        print("")
        
        return indices.flatten()[1:] + 1, distances.flatten()[1:]

    def get_movie_recommendations(self, user_id, n_similar_users=5, n_movies=10):
        """
        Get movie recommendations for a user.
        
        Parameters:
        -----------
        user_id : int
            User ID to get recommendations for
        n_similar_users : int
            Number of similar users to consider
        n_movies : int
            Number of movies to recommend
            
        Returns:
        --------
        list : Recommended movies
        """
        # Get similar users and their distances
        similar_user_list, distance_list = self.get_similar_users(user_id, n_similar_users)
        
        # Calculate weightage for each similar user
        weightage_list = distance_list/np.sum(distance_list)
        mov_rtngs_sim_users = self.user_to_movie_df.values[similar_user_list-1]
        weightage_list = weightage_list[:, np.newaxis] + np.zeros(len(self.movies_list))
        
        # Calculate weighted ratings
        new_rating_matrix = weightage_list * mov_rtngs_sim_users
        mean_rating_list = new_rating_matrix.sum(axis=0)
        
        print("")
        print("Movies recommended based on similar users are:")
        print("")
        
        return self._filter_recommendations(user_id, mean_rating_list, n_movies)

    def _filter_recommendations(self, user_id, mean_rating_list, n_movies):
        """
        Filter and return final movie recommendations.
        
        Parameters:
        -----------
        user_id : int
            User ID to filter movies for
        mean_rating_list : numpy.array
            List of mean ratings for all movies
        n_movies : int
            Number of movies to recommend
            
        Returns:
        --------
        list : Filtered movie recommendations
        """
        # Find last zero rating index
        first_zero_index = np.where(mean_rating_list == 0)[0][-1]
        
        # Sort movies by rating
        sortd_index = np.argsort(mean_rating_list)[::-1]
        sortd_index = sortd_index[:list(sortd_index).index(first_zero_index)]
        
        # Get movies already watched by user
        movies_watched = self.get_user_movies(user_id)
        
        # Filter movies
        filtered_movie_list = list(self.movies_list[sortd_index])
        final_movie_list = []
        count = 0
        
        for movie in filtered_movie_list:
            if movie not in movies_watched:
                count += 1
                final_movie_list.append(movie)
                if count == n_movies:
                    break
        
        if count == 0:
            print("There are no movies left which are not seen by the input users and seen by similar users. "
                  "May be increasing the number of similar users who are to be considered may give a chance "
                  "of suggesting an unseen good movie.")
        else:
            pprint(final_movie_list)
            
        return final_movie_list
    
    def get_similar_movies(self, movie, n = 10):
        ## input to this function is the movie and number of top similar movies you want.
        ## input to this function is the movie and number of top similar movies you want.
        movies_list = list(self.movie_to_user_df.index)
        movie_dict = {movie : index for index, movie in enumerate(movies_list)}

        index = movie_dict[movie]
        knn_input = np.asarray([self.movie_to_user_df.values[index]])
        n = min(len(movies_list)-1,n)
        distances, indices = self.knn_movie_model.kneighbors(knn_input, n_neighbors=n+1)
        
        print("Top",n,"movies which are very much similar to the Movie-",movie, "are: ")
        print(" ")
        movies_list_ret = []
        for i in range(1,len(distances[0])):
            movies_list_ret.append(movies_list[indices[0][i]])
        return movies_list_ret
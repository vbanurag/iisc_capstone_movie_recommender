class MovieFinder:
    def __init__(self, movies_list, case_insensitive_movies_list):
        self.movies_list = movies_list
        self.case_insensitive_movies_list = case_insensitive_movies_list

    def get_possible_movies(self, movie_query):
        """Find possible movie matches based on partial input."""
        temp = ''
        possible_movies = self.case_insensitive_movies_list.copy()
        
        for char in movie_query.lower():
            out = []
            temp += char
            for movie in possible_movies:
                if temp in movie:
                    out.append(movie)
            if len(out) == 0:
                return possible_movies
            out.sort()
            possible_movies = out.copy()
        
        return possible_movies

    def find_exact_movie(self, movie_name):
        """Find exact movie match."""
        movie_name_lower = movie_name.lower()
        try:
            idx = self.case_insensitive_movies_list.index(movie_name_lower)
            return self.movies_list[idx]
        except ValueError:
            return None

    def get_movie_suggestions(self, movie_query):
        """Get movie suggestions based on partial input."""
        possible_movies = self.get_possible_movies(movie_query)
        
        if len(possible_movies) == len(self.movies_list):
            return None
        
        suggestions = []
        for movie in possible_movies:
            idx = self.case_insensitive_movies_list.index(movie)
            suggestions.append(self.movies_list[idx])
            
        return suggestions
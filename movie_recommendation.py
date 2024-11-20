import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from movies_recommendation_content_2 import get_cosine_sim3, get_cosine_sim2, jaccard_set
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer


path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
movies = pd.read_csv("tmdb/tmdb-movie-metadata/versions/2/tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb/tmdb-movie-metadata/versions/2/tmdb_5000_credits.csv")

#Data exploration
C= movies['vote_average'].mean()
m= movies['vote_count'].quantile(0.9)
q_movies = movies.copy().loc[movies['vote_count'] >= m]
q_movies = q_movies.copy().loc[q_movies['original_language']=='en']
#print(q_movies.shape)  & [movies['original_language']=='en']]
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)


print(q_movies[['title', 'vote_count', 'vote_average', 'score','popularity']].head(10))
plt.scatter(q_movies['score'],q_movies['popularity']/100)
plt.show()

#Data Preprocessing

movies=movies.merge(credits,on='title')
movies_sel=movies[[ 'id','genres', 'keywords', 'title', 'overview', 'cast', 'crew','vote_average']]
movies_sel['title_lower']=movies_sel['title'].apply(lambda x: x.lower())
movies_sel['english_language']=movies['original_language']=='en'
movies.dropna(inplace=True)
movies_sel['overview']=movies_sel['overview'].fillna('')
movies_sel['year']=movies['release_date'][0][0:4]


#Vectorization

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_sel['overview'])
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies_sel.index, index=movies_sel['title']).drop_duplicates()
indices_lower = pd.Series(movies_sel.index, index=movies_sel['title_lower']).drop_duplicates()


def get_recommendations(title,len_recommended_movies , cosine_sim2):
        
    idx = indices_lower[title]

    sim_scores = list(enumerate(cosine_sim2[idx]))
       
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    sim_scores = sim_scores[1:len_recommended_movies+1]
  
    movie_indices = [i[0] for i in sim_scores]
    tags=[]
    for i in range(len_recommended_movies):
        
        tags.append(movies_sel['title'].iloc[movie_indices[i]])
    # Return the topmost similar movies
    movies_recommended= movies_sel['title'].iloc[movie_indices]

    return     tags





import requests
#import requests_cache
import json


def get_movies_from_tastedive(movie_name,num_recommended_movies):      
    dest_url = "https://tastedive.com/api/similar"
   
    dp = {'q': movie_name, 'type': 'movie', 'k':'1039874-Moviepro-A6E10BE0','limit':2*num_recommended_movies}
    resp = requests.get(dest_url, params = dp)
    recommendation = resp.json()
    similar=recommendation['similar']
    
    tags=[]
    for i in range(2*num_recommended_movies):
        
        if similar['results'][i]['name'] in list(movies_sel['title']):
            tags.append((similar['results'][i]['name']))
        
    if (len(tags)) > num_recommended_movies:
        return tags[0:num_recommended_movies]
    else:
        return tags


def recommended_movies():
    #movie_name=input('Suggest a English movie name: ')
    num_recommended_movies=5
    for movie_name in ['Quantum of Solace', 'Harry Potter and the chamber of secrets','The Dark knight','Toy Story','The Avengers','Titanic','Avatar']:
        if not movie_name:
            print('Model Recommendation')
            print((q_movies[['title']].head(10)))
        else:
            if movie_name.lower() in list(movies_sel['title_lower']):
                print('Thank You')
            else:
                print('Sorry, we do not have this in our database.')
                return 
            Benchmark_recommendation = get_movies_from_tastedive(movie_name.lower(),num_recommended_movies)

            #print(recommendation.len())
            Model_recommendation=get_recommendations(movie_name.lower(),len(Benchmark_recommendation),cosine_sim)
            cosine_sim3=get_cosine_sim3()
            #cosine_sim2=get_cosine_sim2()
            #Model_recommendation2=get_recommendations(movie_name.lower(),len(Benchmark_recommendation),cosine_sim+cosine_sim2)
            Model_recommendation3=get_recommendations(movie_name.lower(),len(Benchmark_recommendation),cosine_sim+cosine_sim3)
            Accuracy=jaccard_set(Benchmark_recommendation,Model_recommendation)
            #Accuracy2=jaccard_set(Benchmark_recommendation,Model_recommendation2)
            Accuracy3=jaccard_set(Benchmark_recommendation,Model_recommendation3)
            
            print(Accuracy3)
            
recommended_movies()
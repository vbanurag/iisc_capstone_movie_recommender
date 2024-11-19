import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from movies_recommendation_content_2 import get_cosine_sim3
# Download latest version
path = kagglehub.dataset_download("yusufdelikkaya/imdb-movie-dataset")
#print("Path to dataset files:", path)
import warnings

warnings.filterwarnings("ignore")
import pandas as pd 
num_recommended_movies=10
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

movies = pd.read_csv("tmdb/tmdb-movie-metadata/versions/2/tmdb_5000_movies.csv")

credits = pd.read_csv("tmdb/tmdb-movie-metadata/versions/2/tmdb_5000_credits.csv")
#df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
#df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

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
#print(q_movies[['title', 'vote_count', 'vote_average', 'score','popularity']].head(10))
#plt.scatter(q_movies['score'],q_movies['popularity']/100)
#plt.show()
movies=movies.merge(credits,on='title')
movies_sel=movies[[ 'id','genres', 'keywords', 'title', 'overview', 'cast', 'crew','vote_average']]
movies_sel['title_lower']=movies_sel['title'].apply(lambda x: x.lower())
movies_sel['english_language']=movies['original_language']=='en'
movies.dropna(inplace=True)
#print(movies.isnull().sum())


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
movies_sel['overview']=movies_sel['overview'].fillna('')

import nltk
#nltk.download('all')
from nltk import word_tokenize, pos_tag

def extract_nouns(text):
    nouns = []
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        if tag.startswith('NN'):  # NN, NNS, NNP, NNPS are noun tags
            nouns.append(word)
    return str(nouns)


#text = "The cat sat on the mat."
#nouns = extract_nouns(movies_sel['overview'][0])
#print(type(movies_sel['overview'][0]))
#print(type(str(nouns)))  # Output: ['cat', 'mat']

#movies_sel['overview_short']=movies_sel['overview']
#for i in range(len(movies_sel['overview'])):
#    movies_sel['overview_short'][i]=extract_nouns(movies_sel['overview'][i])
#movies_sel['year']=movies['release_date']
#print( movies_sel['overview_short'].head())
#movies_sel['overview']=movies_sel['overview'].apply(lambda x:x.split())
tfidf_matrix = tfidf.fit_transform(movies_sel['overview'])
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_sel.index, index=movies_sel['title']).drop_duplicates()
indices_lower = pd.Series(movies_sel.index, index=movies_sel['title_lower']).drop_duplicates()
movies_sel['year']=movies['release_date'][0][0:4]



#for feature in features:
#    movies_sel[feature] = movies_sel[feature].apply(clean_data)


def get_recommendations(title,len_recommended_movies , cosine_sim2):
    # Get the index of the movie that matches the title
     
    idx = indices_lower[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))
    
    #print(len(movies_sel['vote_average']))
    #sim_scores=sim_scores*movies_sel['vote_average']
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
    sim_scores = sim_scores[1:len_recommended_movies+1]
    print((sim_scores))
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    tags=[]
    for i in range(num_recommended_movies):
        #print(movies_sel['title'].iloc[movie_indices[i]])
        tags.append(movies_sel['title'].iloc[movie_indices[i]])
    # Return the top 10 most similar movies
    movies_recommended= movies_sel['title'].iloc[movie_indices]
    #movies_recommended['year']= movies_sel['year'].iloc[movie_indices]
    #print(type(movies_recommended))
    return     tags
#= movies_sel['title'].iloc[movie_indices]

#movies_sel['year'].iloc[movie_indices]


import requests
#import requests_cache
import json

#def get_movies_from_tastedive(movie_name):      
#    dest_url = "https://tastedive.com/api/similar"
   
#    dp = {'q': movie_name, 'type': 'movie', 'k':'1039874-Moviepro-A6E10BE0','limit':10}
#    resp = requests.get(dest_url, params = dp)

#    return resp.json()
#print(get_movies_from_tastedive('Titanic'))
def get_movies_from_tastedive(movie_name):      
    dest_url = "https://tastedive.com/api/similar"
   
    dp = {'q': movie_name, 'type': 'movie', 'k':'1039874-Moviepro-A6E10BE0','limit':2*num_recommended_movies}
    resp = requests.get(dest_url, params = dp)
    recommendation = resp.json()
    similar=recommendation['similar']
    #recommendation2=pd.DataFrame.from_dict(similar['results'])
    tags=[]
    for i in range(2*num_recommended_movies):
        
        if similar['results'][i]['name'] in list(movies_sel['title']):
            tags.append((similar['results'][i]['name']))
        #else:
            #print('Movie not found')
    if (len(tags)) > num_recommended_movies:
        return tags[0:num_recommended_movies]
    else:
        return tags
#print(type(get_movies_from_tastedive('Titanic')))

def recommended_movies():
    movie_name=input('Suggest a English movie name: ')

    if movie_name.lower() in list(movies_sel['title_lower']):
        print('Thank You')
    else:
        print('Sorry, we do not have this in our database.')
        return 
    Benchmark_recommendation = get_movies_from_tastedive(movie_name.lower())

    #print(recommendation.len())
    Model_recommendation=get_recommendations(movie_name.lower(),len(Benchmark_recommendation),cosine_sim)
    cosine_sim3=get_cosine_sim3()
    Model_recommendation2=get_recommendations(movie_name.lower(),len(Benchmark_recommendation),cosine_sim3)
    final_list = list(set(Benchmark_recommendation) | set(Model_recommendation))
    Accuracy=(2-len(final_list)/len(Benchmark_recommendation))*100
    print("Benchmark Recommendation")
    print((Benchmark_recommendation))
    print('Model Recommendation')
    print((Model_recommendation))
   # print((Model_recommendation))
    print(Accuracy)

recommended_movies()
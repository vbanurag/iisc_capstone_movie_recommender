import numpy as np
import matplotlib.pyplot as plt
import kagglehub
# Download latest version
#path = kagglehub.dataset_download("yusufdelikkaya/imdb-movie-dataset")
#print("Path to dataset files:", path)
import warnings

warnings.filterwarnings("ignore")
import pandas as pd 

#path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

movies = pd.read_csv("tmdb/tmdb-movie-metadata/versions/2/tmdb_5000_movies.csv")

credits = pd.read_csv("tmdb/tmdb-movie-metadata/versions/2/tmdb_5000_credits.csv")
movies=movies.merge(credits,on='title')
movies_sel=movies[[ 'id','genres', 'keywords', 'title', 'overview', 'cast', 'crew','vote_average','production_companies']]
movies_sel['english_language']=movies['original_language']=='en'
movies.dropna(inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
movies_sel['overview']=movies_sel['overview'].fillna('')
movies_sel['overview']=movies_sel['overview'].apply(lambda x:x.split())

import ast
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres','production_companies']
for feature in features:
    movies_sel[feature] = movies_sel[feature].apply(literal_eval)
def convert5(obj):
    tags=[]
    iterator=0
    for i in ast.literal_eval(obj):
        if iterator!=5:   #top three in the cast
            tags.append(i["name"])
            iterator+=1
        else:
            break
    return tags

def convert(obj):
    tags=[]
    for i in ast.literal_eval(obj): #since every genres column is a string of list we conver it into list and the iterates
        tags.append(i["name"])
    return tags
def director(obj): #director extraction from crew
    tags=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':   #Since only 1 director in movie
            tags.append(i["name"]) 
            break
    return tags
def get_director(x):
    for item in x:
        if item['job'] == 'Director':
            return item['name']
    return np.nan
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

movies_sel['crew']=movies['crew'].apply(director)
movies_sel['cast']=movies['cast'].apply(convert5)
movies_sel['genres']=movies['genres'].apply(convert)

movies_sel['keywords']=movies['keywords'].apply(convert)
movies_sel['production_companies']=movies['production_companies'].apply(convert)


features = ['cast',  'crew', 'genres','keywords','title','production_companies']
for feature in features:
    movies_sel[feature] = movies_sel[feature].apply(clean_data)


movies_sel["combo_list"] =   movies_sel["genres"]+ movies_sel["keywords"]+ movies_sel["cast"]+ movies_sel["crew"]#+movies_sel['production_companies']

movies_sel["combo_list"] =movies_sel["combo_list"].apply(lambda x: " ".join((x)))

movies_sel["combo_list"] =movies_sel["combo_list"].apply(lambda x: x.lower())


movies_sel['combo_list']=movies_sel['combo_list'].fillna('')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
tfidf = TfidfVectorizer(stop_words='english')
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies_sel['combo_list'])

tfidf_matrix = tfidf.fit_transform(movies_sel["combo_list"])


# Compute the cosine similarity matrix
cosine_sim3 = linear_kernel(tfidf_matrix, tfidf_matrix)
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

def jaccard_set(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def get_cosine_sim3():
    return  cosine_sim3
def get_cosine_sim2():
    return  cosine_sim2

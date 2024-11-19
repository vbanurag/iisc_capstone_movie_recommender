import requests
#import requests_cache
import json
import pandas as pd

def get_movies_from_tastedive(movie_name):      
    dest_url = "https://tastedive.com/api/similar"
   
    dp = {'q': movie_name, 'type': 'movie', 'k':'1039874-Moviepro-A6E10BE0','limit':5}
    resp = requests.get(dest_url, params = dp)
    recommendation = resp.json()
    similar=recommendation['similar']
    #recommendation2=pd.DataFrame.from_dict(similar['results'])
    tags=[]
    for i in range(5):
        tags.append((similar['results'][i]['name']))
    return tags
#print(type(get_movies_from_tastedive('Titanic')))

def recommended_movies():
    movie_name=input('Suggest a English movie name: ')
    recommendation = get_movies_from_tastedive(movie_name)
    print(recommendation)
recommended_movies()
#print((recommendation))

#tags=recommendation.apply(convert3)
#similar=recommendation['similar']
#recommendation2=pd.DataFrame.from_dict(similar['results'])
#tags=[]
#for i in range(5):
#    tags.append((similar['results'][i]['name']))
#print(tags)
#print(recommendation2.apply(convert3))



# Apply the function to the 'Text' column
 #recommendation['results'].apply(extract_name)

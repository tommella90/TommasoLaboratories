import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from random import randint
import parse


#%% OPEN HTML URL
url = "https://www.billboard.com/charts/hot-100/"
response = requests.get(url)
response.status_code
print(response.content)
soup = bs(response.content)
soup

soup.title.get_text()       ## title
soup.find("p").get_text()   ## all paragraphs
soup.find_all("a")          ## all elements of a tag

#%% soup select
'''
mylist = soup.select(".o-chart-results-list-row-container")
mylist[0].get_text()

mylist2 = soup.select(".o-chart-results-list-row")
mylist2[0].get_text()

mylist3 = soup.select(".o-chart-results-list__item h3")
mylist3[0].get_text(strip=True)

soup.find(id='title-of-a-story')

mylist4 = soup.select(".c-title.a-no-trucate")
mylist4[0].get_text(strip=True)

mylist5 = soup.select(".c-label.a-no-trucate")
mylist5[0].get_text(strip=True)
'''

#%% CREATE DATAFRAME WITH ARTISTS AND NAMES
artists = []
songs = []
pos = []
max_pos = []

leng = len(soup.select(".c-title.a-no-trucate"))
soup.select(".c-label.a-font-primary-m")[0].get_text(strip=True)

for i in range (0,leng):
    song2 = str(soup.select(".c-title.a-no-trucate")[i].get_text(strip=True))
    print(song2)
    songs.append(song2)

for i in range (0,leng):
    artist2 = str(soup.select(".c-label.a-no-trucate")[i].get_text(strip=True))
    print(artist2)
    artists.append(artist2)

'''
for i in range(0,leng):
    pos2 = str(soup.select(".c-label.a-font-primary-m")[i].get_text(strip=True))
    print(pos2)
    pos.append(pos2)

for i in range(0,leng):
    max_pos2 = str(soup.select(".o-chart-results-list__item")[i].get_text(strip=True))
    print(max_pos2)
    max_pos.append(max_pos2)

len(soup.select(".u-max-width-960 .chart-results-list .o-chart-results-list-row-container .o-chart-results-list-row .lrv-u-width-100p .o-chart-results-list__item .c-label.a-font-primary-m"))
'''
song_artis = dict(zip(artists, songs))

df = pd.DataFrame(
    {'Artists': artists,
     'Songs': songs,
     #'Wks on Chart': pos,
     #'Max position': max_pos,
     })

print(df)

df.to_csv('C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/data/labs/songs_df.csv', index = False, encoding='utf-8') # False: not include index
print('s')

#%% SONG RECCOMENDATION
#Desesperados
import pandas as pd
from termcolor import colored
from random import randint

df = pd.read_csv("C:/Users/tomma/Documents/data_science/berlin/TommasoLaboratories/data/labs/songs_df.csv")

select_song = input('Please enter your song: ')

if select_song.lower() in str(list(df["Songs"])).lower():
    r = randint(0, len(df.index))
    print("Other song recommended: ", df.iloc[r, 1])
else:
    print(colored("THE SONG SELECTED IS NOT IN THE DATABASE", attrs=['bold']))
    #print(colored("THE SONG SELECTED IS NOT IN THE DATABASE"), colored('red'))

#%%
import re
text = str(soup)
class_html = "c-label  a-font-primary-m lrv-u-padding-tb-050@mobile-max"
pattern = re.compile(r'tb-050@mobile-max">\n\t\n\t(.*)(.*)')
#pattern = re.compile(r'tb-050@mobile-max">\n\t\n\t\()')
matches = pattern.finditer(text)
match = []
for i in matches:
    print(i)
    match.append(i)
match

#%%
list2 = []
for i in match:
    list2.append(str(i)[-4:-1])
list2

#%%
list2[0:20]
#%%

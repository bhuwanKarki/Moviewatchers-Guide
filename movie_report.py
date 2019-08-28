#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:40:38 2019

@author: bhuwankarki
"""

#import the required libraries
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import requests, zipfile, io
from collections import Counter
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fpdf import FPDF

#get the file from the http server
r = requests.get("http://files.grouplens.org/datasets/movielens/ml-1m.zip" )
# read the zip file and extract the zipfile in the folder
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# Specify User's Age and Occupation Column
AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 
        35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 
               15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 
               19: "unemployed", 20: "writer" }

# columns names for the movies.dat , rating.dat ,user.data files
ratings_names=["UserId","MovieID","Rating","Timestamp"]
user_names=["UserId","Gender","Age","Occupation","Zip-code"]
movies_name=["MovieID","Title","Genres"]


#load the dat file into a pandas dataframe
movies=pd.read_csv("ml-1m/movies.dat",sep="::",engine="python",header=None,names=movies_name)
#extract the year from the title column with regrex
movies['year'] = movies.Title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year
# remove the year in the Title
movies["Title"]=movies["Title"].apply(lambda x:re.sub(r'\([^)]*\)', '',x))

#split the genres into a single list 
genres_unique = pd.DataFrame(movies.Genres.str.split('|').tolist()).stack().unique()
 # Format into DataFrame to store later
genres_unique = pd.DataFrame(genres_unique, columns=['Genre'])
#create a new data frame with genres creating a dummy variable (one-hot encoding)
new_movies = movies.join(movies.Genres.str.get_dummies().astype(bool))
new_movies.drop('Genres', inplace=True, axis=1)
new_movies.dropna(inplace=True)

#movies made per year
movies_made=movies[["MovieID","year"]].groupby("year")
plt.figure(figsize=(10,10))
plt.title("Movies Made Per Year")
plt.ylabel('Number_of_movies')
plt.xlabel('Year');
sns.lineplot(movies_made.year.first(),movies_made.MovieID.nunique(),sizes=(.25, 2.5))
plt.savefig("movie_per_year.png")

#create a counter 
genreCount = Counter()
for row in movies.itertuples():
    genreCount.update(row[3].split("|")) 
c=dict(genreCount)
keyword=[]
for k,v in c.items():
    keyword.append(([k,v]))
keyword.sort(key = lambda x:x[1], reverse = True)

# Define the dictionary used to produce the genre wordcloud
genres = dict()
trunc_occurences = keyword[0:18]
for s in trunc_occurences:
    genres[s[0]] = s[1]

# Create the wordcloud
genre_wordcloud = WordCloud(width=700,height=400, background_color='white')
genre_wordcloud.generate_from_frequencies(genres)

# Plot the wordcloud
f, ax = plt.subplots(figsize=(16, 8))

plt.imshow(genre_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.savefig("wordcloud.png")

plt.figure(figsize=(5,1))
df = pd.DataFrame({'All_movies' : movies_made.MovieID.nunique().cumsum()})
# Plot histogram for each individual genre
for genre in genres_unique.Genre:
    dftmp = new_movies[new_movies[genre]][['MovieID', 'year']].groupby('year')
    df[genre]=dftmp.MovieID.nunique().cumsum()
df.fillna(method='ffill', inplace=True)
df.loc[:,df.columns!='All_movies'].plot.area(stacked=True, figsize=(10,5))
# Plot histogram for all movies
plt.plot(df['All_movies'], marker='o', markerfacecolor='black')
plt.xlabel('Year')
plt.ylabel('Cumulative number of movies-genre')
plt.title('Total movies-genre') # Many movies have multiple genres, so counthere is higher than number of movies
plt.legend(loc=(1.05,0), ncol=1)
plt.savefig("total_movie_genre.png",bbox_inches="tight")


# Plot simple scatter of the number of movies tagged with each genre
plt.figure(figsize=(10,5))
barlist = df.iloc[-1].plot.bar()
barlist.patches[0].set_color('r') # Color 'All_movies' differently, as it's not a genre tag count
plt.xticks(rotation='vertical')
plt.title('Movies per genre tag')
plt.xlabel('Genre')
plt.ylabel('Number of movies tagged')
plt.savefig("movie_per_genre.png")


#Rating dataframe Analysis

rating=pd.read_csv("ml-1m/ratings.dat",sep="::",header=None,engine="python",
                   names=ratings_names)
rating["Timestamp"]=pd.to_datetime(rating["Timestamp"],infer_datetime_format=True)
rating.Timestamp = rating.Timestamp.dt.year

dftmp = rating[['MovieID','Rating']].groupby('MovieID').mean()


# Plot general histogram of all ratings
dftmp.hist(bins=25, grid=False, edgecolor='b',density=True, label ='All genres', figsize=(10,5))
# Plot histograms (kde lines for better visibility) per genre
for genre in genres_unique.Genre:
    dftmp = new_movies[new_movies[genre]==True]
    dftmp = rating[rating.set_index('MovieID').index.isin(dftmp.set_index('MovieID').index)]
    dftmp = dftmp[['MovieID','Rating']].groupby('MovieID').mean()
    dftmp.Rating.plot(grid=False, alpha=0.6, kind='kde', label=genre)
    avg = dftmp.Rating.mean()
    std = dftmp.Rating.std()
    
plt.legend(loc=(1.05,0), ncol=2)
plt.xlim(0,5)
plt.xlabel('Movie rating')
plt.title('Movie rating histograms')
plt.savefig("movie_rating_hist.png",bbox_inches="tight")


dftmp = rating[['UserId','Rating']].groupby('UserId').mean()
# Plot histogram
dftmp.plot(kind='hist', bins=50, grid=0, density=True, edgecolor='black', figsize=(10,5))
# evaluate the histogram
values, base = np.histogram(dftmp, bins=40, density=True)
# evaluate the cumulative (multiply by the average distance between points in the x-axis to get UNIT area)
cumulative = np.cumsum(values) * np.diff(base).mean()
# plot the cumulative function
plt.plot(base[:-1], cumulative, c='blue', label='CDF')
plt.xlim(0,5)
plt.legend()
plt.xlabel ('Average movie rating')
plt.ylabel ('Normalized frequency')
plt.title ('Average ratings per user')
plt.savefig("Avg_rating_per_user.png")


dftmp = rating[['UserId', 'MovieID']].groupby('UserId').count()
dftmp.columns=['num_ratings']
plt.figure(figsize=(15,5))
plt.scatter(dftmp.index, dftmp.num_ratings, edgecolor='black')
plt.xlim(0,len(dftmp.index))
plt.ylim(0,)
plt.title('Ratings per user')
plt.xlabel('userId')
plt.ylabel('Number of ratings given')
plt.savefig("rating_per_user.png")

# Histogram of ratings counts.
plt.figure(figsize=(15,5))
plt.hist(dftmp.num_ratings, bins=100, edgecolor='black', log=True)
plt.title('Ratings per user')
plt.xlabel('Number of ratings given')
plt.ylabel('Number of userIds')
plt.xlim(0,)
plt.xticks(np.arange(0,3000,200))
plt.savefig("rating_user_id.png")


dftmp = rating[['UserId', 'MovieID']].groupby('MovieID').count()
dftmp.columns=['num_ratings']
plt.figure(figsize=(15,5))
plt.scatter(dftmp.index, dftmp.num_ratings, edgecolor='black')
plt.xlim(0,dftmp.index.max())
plt.ylim(0,)
plt.title('Ratings per movie')
plt.xlabel('movieId')
plt.ylabel('Number of ratings received')
plt.savefig("rating_per_movie.png")

# Histogram of ratings counts.
plt.figure(figsize=(15,5))
plt.hist(dftmp.num_ratings, bins=100, edgecolor='black', log=True)
plt.title('Ratings per movie')
plt.xlabel('Number of ratings received')
plt.ylabel('Number of movieIds')
plt.xlim(0,)
plt.savefig("rating_per_movieID.png")


# Which is the best most popular movie ever??
tmp = rating.set_index('MovieID').loc[dftmp.index[dftmp.num_ratings>1000]].groupby('MovieID').mean()
best = movies.set_index('MovieID').loc[tmp.Rating.idxmax].Title

#user.dat dataframe analysis

user=pd.read_table("ml-1m/users.dat",sep="::",header=None,names=user_names,engine="python")

user["age_desc"]=user["Age"].apply(lambda x:AGES[x])
user["occ_desc"]=user["Occupation"].apply(lambda x:OCCUPATIONS[x])
new_user=user.drop(columns=["Age","Occupation"])
plt.figure(figsize=(15,5))
plt.title("Distribution of Users' Ages")
plt.ylabel('Number of Users')
plt.xlabel('Age')
sns.distplot(user.Age,bins=20)
plt.savefig("user_age.png")

new_data=pd.merge(pd.merge(rating,user),movies)

#countplot for rating
plt.figure(figsize=(15,5))
plt.title("Movie count on the basis of rating for gender")
sns.countplot(x='Rating',data=new_data,hue='Gender',palette='coolwarm')
plt.savefig("Movie_count_r_g.png")

plt.figure(figsize=(15,5))
plt.title("Movie count on the basis of Age for gender")
sns.countplot(x="Age",hue='Gender',data=new_data,palette='coolwarm',)
plt.savefig("Movie_count_A_g.png")

#finding mean rating of all the movies
mean_rating=new_data.groupby('Title')['Rating'].agg('mean')

plt.figure(figsize=(15,5))
plt.title("top 25 movies on the Basis of Rating")
plot1=sns.barplot(x=mean_rating.nlargest(25).index,y=mean_rating.nlargest(25).values)
plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)
plt.savefig("top_25_movie.png",bbox_inches="tight")

#content-based Recommendation Model
movies["Genres"]=movies["Genres"].str.split('|')
movies['Genres'] = movies['Genres'].fillna("").astype('str')

# TfidfVectorizer function from scikit-learn, which transforms text to feature vectors that can be used as input to estimator.

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['Genres'])

#we will use sklearn's linear_kernel instead of cosine_similarities since it is much faster.

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)#alculate a numeric quantity that denotes the similarity between two movies

 #Build a 1-dimensional array with movie titles
titles = movies['Title']
indices = pd.Series(movies.index, index=movies['Title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

recommend1=genre_recommendations('Good Will Hunting ').head(20)
recommend1=pd.DataFrame(recommend1)

recommend2=genre_recommendations('Saving Private Ryan ').head(20)
recommend2=pd.DataFrame(recommend2)




 
# Create instance of FPDF class
# Letter size paper, use inches as unit of measure
pdf=FPDF(format='A4')
 
# Add new page. Without this you cannot create the document.
pdf.add_page()
 
# Remember to always put one of these at least once.
pdf.set_font('Times','',10.0) 
 
# Effective page width, or just epw
epw = pdf.w - 2*pdf.l_margin
 
# Set column width to 1/4 of effective page width to distribute content 
# evenly across table and page
col_width = epw/4
 
 
# Text height is the same as current font size
th = pdf.font_size
 
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'Movie lens data Analysis', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)

# Line break equivalent to 4 lines
pdf.ln(4*th)
 
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'Movies Table', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
 
pdf.ln(4*th)
# Here we add more padding by passing 2*th as height
for row in [movies.columns.values.tolist()]+movies[:10].values.tolist():
    for datum in row:
        # Enter data in colums
        pdf.cell(col_width, 2*th, str(datum), border=1)
 
    pdf.ln(2*th)
    

# Line break equivalent to 4 lines
pdf.ln(2*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'Movies Table Analysis', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
pdf.ln(4*th)
pdf.image('movie_per_year.png', x = None, y = None, w = 200, h = 100, type = '', link = '')
pdf.ln(2*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'Movies per Genres', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
pdf.ln(4*th)
pdf.image('wordcloud.png', x = None, y = None, w = 200, h = 100, type = '', link = '')
pdf.image('total_movie_genre.png', x = None, y = None, w = 170, h = 100, type = '', link = '')
pdf.image('movie_per_genre.png', x = None, y = None, w = 200, h = 100, type = '', link = '')

pdf.ln(4*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'Rating Table', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
pdf.ln(4*th)
# Here we add more padding by passing 2*th as height
for row in  [rating.columns.values.tolist()]+rating[50:60].values.tolist():
    for datum in row:
        # Enter data in colums
        pdf.cell(col_width, 2*th, str(datum), border=1)
 
    pdf.ln(2*th)
    
pdf.ln(2*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'Rating Table Analysis', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
pdf.ln(4*th)   
pdf.image('movie_rating_hist.png', x = None, y = None, w = 150, h = 100, type = '', link = '')
pdf.image('Avg_rating_per_user.png', x = None, y = None, w = 200, h = 150, type = '', link = '')
pdf.image('rating_per_user.png', x = None, y = None, w = 200, h = 120, type = '', link = '')
pdf.image('rating_user_id.png', x = None, y = None, w = 200, h = 120, type = '', link = '')
pdf.image('rating_per_movie.png', x = None, y = None, w = 200, h = 120, type = '', link = '')
pdf.image('rating_per_movieID.png', x = None, y = None, w = 200, h = 100, type = '', link = '')

 
pdf.add_page()
pdf.ln(4*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'User Table', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
pdf.ln(4*th)
# Here we add more padding by passing 2*th as height
for row in  [new_user.columns.values.tolist()]+new_user[50:60].values.tolist():
    for datum in row:
        # Enter data in colums
        pdf.cell(col_width/1.5, 2*th, str(datum), border=1)
 
    pdf.ln(2*th)
pdf.ln(2*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'user table', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
pdf.ln(4*th)     
pdf.image('user_age.png', x = None, y = None, w = 200, h = 100, type = '', link = '')
pdf.image('Movie_count_r_g.png', x = None, y = None, w = 150, h = 100, type = '', link = '')
pdf.image('Movie_count_A_g.png', x = None, y = None, w = 150, h = 100, type = '', link = '')
pdf.image('top_25_movie.png', x = None, y = None, w = 150, h = 100, type = '', link = '')

pdf.ln(4*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, ' Best movie on the Basis of Numbers of Rating  %s' % (best), align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
pdf.ln(4*th)

 
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'Movies Table', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
 
pdf.ln(4*th)

pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, 'recommendation for movie simiar to good will hunting', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
 
pdf.ln(4*th)
# Here we add more padding by passing 2*th as height
for row in [recommend1.columns.tolist()]+recommend1[:10].values.tolist():
    for datum in row:
        # Enter data in colums
        pdf.cell(col_width, 2*th, str(datum), border=1)
 
    pdf.ln(2*th)
pdf.ln(4*th)
pdf.set_font('Times','B',14.0) 
pdf.cell(epw, 0.0, ' recommendation for movie similar to saving pirate Ryan', align='C')
pdf.set_font('Times','',10.0) 
pdf.ln(0.5)
 
pdf.ln(4*th)

pdf.ln(4*th)
# Here we add more padding by passing 2*th as height
for row in [recommend2.columns.tolist()]+recommend2[:10].values.tolist():
    for datum in row:
        # Enter data in colums
        pdf.cell(col_width, 2*th, str(datum), border=1)
 
    pdf.ln(2*th)


 
pdf.output('test.pdf', 'F')
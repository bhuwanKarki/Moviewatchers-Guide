# Analyzing-MovieLens-1M-Dataset



## Dependencies (pip install):
```
numpy
pandas
matplotlib
seaborn
wordcloud
fpdf
sklearn
```

## DATA PRE-PROCESSING:

**  Desired outcome of the case study.**

-   In this case study we will look at the MovieLens 1M Data Set.
    -   It contains data about users and how the rate movies.
-   The idea is to  _analyze_  the data set, make  _conjectures_, support  those conjectures with  _data_, and  _tell a story_  about the data!
## ANALYSIS:
-Three different dataset are loaded into the pandas dataframe
- The loaded dataframe are filtered and processes to obtain a meaningful information from them
- Graphs are plotted so that visualisation can help to clear the result.


## How to Use:
- The project folder contain a *requirement.txt * file to install the required dependencies
- run the command to install the required dependencies



```python
pip install -r requirement.txt

```
Then Run,
```python
python3 movie_report.py

```


# Basic implementation of Content-Based Recommendation System
## Content-Based Recommendation Model

A content based recommender works with data that the user provides, either explicitly (rating) or implicitly (clicking on a link). Based on that data, a user profile is generated, which is then used to make suggestions to the user. As the user provides more inputs or takes actions on the recommendations, the engine becomes more and more accurate.
The concepts of Term Frequency (TF) and Inverse Document Frequency (IDF) are used in information retrieval systems and also content based filtering mechanisms (such as a content based recommender).
TF is simply the frequency of a word in a document. IDF is the inverse of the document frequency among the whole corpus of documents. 
Below is the equation to calculate the TF-IDF score:
![Result-1](https://github.com/khanhnamle1994/movielens/raw/cb1fe40c99cdd61c3c714e501e11f699c87b0eed/images/tfidf.jpg)

Vector Space Model which computes the proximity based on the angle between the vectors. In this model, each item is stored as a vector of its attributes (which are also vectors) in an n-dimensional space and the angles between the vectors are calculated to determine the similarity between the vectors. Next, the user profile vectors are also created based on his actions on previous attributes of items and the similarity between an item and a user is also determined in a similar way.
![Result-2](http://dataconomy.com/wp-content/uploads/2015/04/Five-most-popular-similarity-measures-implementation-in-python-4-620x475.png)

This genreate the PDF report test.pdf which show the basic analysis of the movie -dataset and the content based recommender

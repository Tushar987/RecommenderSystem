
# coding: utf-8

# # Recommender System
# 
# We will be developing a movie recommender system based upon Collaborative-filtering to predict the name of the movie based upon the reviews of the other critics having similar taste. The systesm uses two different methods for finding similairties between the critics known as Euclidean-Distance-Score and Pearson-Correlation-Score. The final reault for both the methods were almost similar. After finding the similarity between critics, it uses the weighted average method to assign higher weight to the peer interest critics. Finally, It normalizes the score by deviding it by the similarities of the critics who reviewed that movie.
# 
# 
# #### Finding Similar DataPoint
# 
# Two ways for calculating similarity scores : 
# 1. Euclidean Distance Score
# 2. Pearson Correlation Score
# 
# <hr style="height:2px">

# In[ ]:

# Import Packages

from math import sqrt
import pandas as pd
import numpy as np

# Getting more than one output Line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:

# Getting the Dataset

df1= pd.read_csv("../input/movies.csv",usecols=['movieId','title']) # ,nrows=5000)
df1.describe(include='all')
df1.head()

df2=pd.read_csv("../input/ratings.csv",usecols=['userId','movieId','rating']) #,nrows=1000)
df2.describe(include='all')
df2.head()


# > ## **Euclidean Distance Score**
# 
# euclidean distance is the square root of the sum of squared differences between corresponding elements of the two vectors.Euclidean distance is only appropriate for data measured on the same scale.
# * distance = 1/(1+sqrt of sum of squares between two points)
# * value varies between 0 to 1, where closeness to 1 implies higher similarity.****

# In[ ]:

def euclidean_distance(person1,person2):
    #Getting details of person1 and person2
    df_first= df2.loc[df2.userId==person1]
    df_second= df2.loc[df2.userId==person2]
    
    #Finding Similar Movies for person1 & person2 
    ratings= pd.merge(df_first,df_second,how='inner',on='movieId')
    
    #If no similar movie found, return 0 (No Similarity)
    if(len(ratings)==0): return 0
    
    #sum of squared difference between ratings
    sum_of_squares=sum(pow((ratings['rating_x']-ratings['rating_y']),2))
    return 1/(1+sum_of_squares)
    
# Checking working by passing similar ID, Corerelation should be 1
euclidean_distance(1,1) # Swwweeettt!!!


# > ## **Pearson Correlation Score** 
# 
# * Correlation between sets of data is a measure of how well they are related. It shows the linear relationship between two sets of data. In simple terms, it answers the question, Can I draw a line graph to represent the data?
# 
# * * Value varies between -1 to 1.[ 0-> Not related ; -1 -> perfect negatively corelated ; 1-> perfect positively corelated] 
# 
# Slightly better than Euclidean because it addresses the the situation where the data isn't normalised. Like a User is giving high movie ratings in comparison to AVERAGE user.

# In[ ]:

def pearson_score(person1,person2):
    
    #Get detail for Person1 and Person2
    df_first= df2.loc[df2.userId==person1]
    df_second= df2.loc[df2.userId==person2]
    
    # Getting mutually rated items    
    ratings= pd.merge(df_first,df_second,how='inner',on='movieId')
    
    # If no rating in common
    n=len(ratings)
    if n==0: return 0

    #Adding up all the ratings
    sum1=sum(ratings['rating_x'])
    sum2=sum(ratings['rating_y'])
    
    ##Summing up squares of ratings
    sum1_square= sum(pow(ratings['rating_x'],2))
    sum2_square= sum(pow(ratings['rating_y'],2))
    
    # sum of products
    product_sum= sum(ratings['rating_x']*ratings['rating_y'])
    
    ## Calculating Pearson Score
    numerator= product_sum - (sum1*sum2/n)
    denominator=sqrt((sum1_square- pow(sum1,2)/n) * (sum2_square - pow(sum2,2)/n))
    if denominator==0: return 0
    
    r=numerator/denominator
    
    return r

#Checking function by passing similar ID, Output should be 1
pearson_score(1,1)


# >## **Getting result based on Pearson Score**

# In[ ]:

# Returns the best matches for person from the prefs dictionary.
# Number of results and similarity function are optional params.
def topMatches(personId,n=5,similarity=pearson_score):
    scores=[(similarity(personId,other),other) for other in df2.loc[df2['userId']!=personId]['userId']]
    # Sort the list so the highest scores appear at the top
    scores.sort( )
    scores.reverse( )
    return scores[0:n]

topMatches(1,n=3) ## Getting 3 most similar Users for Example 


# >## **Getting Recommendation**

# In[ ]:

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommnedation(personId, similarity=pearson_score):
    '''
    totals: Dictionary containing sum of product of Movie Ratings by other user multiplied by weight(similarity)
    simSums: Dictionary containung sum of weights for all the users who have rated that particular movie.
    '''
    totals,simSums= {},{}
    
    df_person= df2.loc[df2.userId==personId]
    
    for otherId in df2.loc[df2['userId']!=personId]['userId']: # all the UserID except personID
        
        # Getting Similarity with OtherID
        sim=similarity(personId,otherId)
        
        # Ignores Score of Zero or Negatie correlation         
        if sim<=0: continue
            
        df_other=df2.loc[df2.userId==otherId]
        
        #Movies not seen by the personID
        movie=df_other[~df_other.isin(df_person).all(1)]
        
        for movieid,rating in (np.array(movie[['movieId','rating']])):
            #similarity* Score
            totals.setdefault(movieid,0)
            totals[movieid]+=rating*sim
            
            #Sum of Similarities
            simSums.setdefault(movieid,0)
            simSums[movieid]+=sim
            
        
        
        
        # Creating Normalized List
        ranking=[(t/simSums[item],item) for item,t in totals.items()]
        
        # return the sorted List
        ranking.sort()
        ranking.reverse()
        recommendedId=np.array([x[1] for x in ranking])
        
        
        return np.array(df1[df1['movieId'].isin(recommendedId)]['title'])[:20]


# In[ ]:

# Example Recoomendation
#returns 20 recommended movie for the given UserID
# userId can be ranged from 1 to 671
getRecommnedation(1)
getRecommnedation(671)


# 
# ### ***Future Improvements***
# ---
# * Applying more sophisticated similarities score methods to improve the suggestions.
# * Write a function to precompute user similarities, and alter the recommendation code to use only the top five other users to get recommendations.
# 
# 

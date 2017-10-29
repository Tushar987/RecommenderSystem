# **Recommender System**
---

This repo contains a movie recommender system based upon Collaborative-filtering to predict the name of the movie as per the reviews of the other users having similar taste. The system uses two different methods for finding similarities between the users known as Euclidean-Distance-Score and Pearson-Correlation-Score. The final result for both the methods were almost similar. After finding the similarity between users, it uses the weighted average method to assign a higher weight to the peer interest critics. Finally, It normalizes the score by dividing it by the similarities of the users who reviewed that movie.


##### **Finding Similar DataPoint**

Two ways for calculating similarity scores
1. Euclidean Distance Score
2. Pearson Correlation Score

### ***Specification***
---
* **Environment** : Jupyter Notebook
* **Language** : Python3
* **Dataset** : [The MovieLens Dataset]( https://grouplens.org/datasets/ "Grouplens.org")

### ***Future Improvements***
---
* Applying more sophisticated similarities score methods to improve the suggestions.
* Write a function to precompute user similarities, and alter the recommendation code to use only the top five other users to get recommendations.




### ***Citation***
***
The MovieLens Dataset :
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=<http://dx.doi.org/10.1145/2827872>


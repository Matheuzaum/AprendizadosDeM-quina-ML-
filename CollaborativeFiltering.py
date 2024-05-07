#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('moviedataset\ml-latest\movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('moviedataset/ml-latest/ratings.csv')

#Head is a function that gets the first N rows of a dataframe. N's default is 5.
print(movies_df.head())

#Let's remove the year from the title column by using pandas' replace function and store it in a new year column.

#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

print("depois de ajustado :")

print(movies_df.head())

#Dropping the genres column
movies_df = movies_df.drop('genres', axis=1)

print("Eliminando mais uma coluna desnecessária : ")

print(movies_df.head())

print("Agora, veja as avaliações dessa coluna : ")

print(ratings_df.head())

#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', axis=1)

print("Tirando uma coluna desnecessária : ")

print(ratings_df.head())

#COMEÇANDO A RECOMENDAR, DE FATO
#Now it's time to start our work on recommendation systems.

userInput = [
            {'title':'Breakfast Club, The (1995)', 'rating':5},
            {'title':'Toy Story (1995)', 'rating':3.5},
            {'title':'Jumanji (1995)', 'rating':2},
            {'title':"Pulp Fiction (1995)", 'rating':5},
            {'title':'Akira (1995)', 'rating':4.5}
         ]

inputMovies = pd.DataFrame(userInput)

print(inputMovies)

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', axis=1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.

#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

print("AÍ VAI")

print(userSubset.head())

#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

#Let's also sort these groups so the users that share the most movies in common with the input have higher priority.
#This provides a richer recommendation since we won't go through every single user.

#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

#Similarity of users to input user

#Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the key is the user Id and the value is the coefficient.

#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')

if not pearsonDF.empty:
    pearsonDF.columns = ['similarityIndex']
    print("ok")

pearsonDF['userId'] = pearsonDF.index

# Remover os parênteses da coluna 'userId'
pearsonDF['userId'] = pearsonDF['userId'].apply(lambda x: x[0])

pearsonDF.index = range(len(pearsonDF))

print("veja o pearsonDF : ")

print(pearsonDF.head())

#The top x similar users to input user

print("Top users : ")

topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
print(topUsers.head())

#Rating of selected users to all movies

print("Top Users Rating : ")

# Converta a coluna 'userId' para o mesmo tipo de dados em ambos os DataFrames
topUsers['userId'] = topUsers['userId'].astype(int)
ratings_df['userId'] = ratings_df['userId'].astype(int)

# Mesclar os DataFrames usando a coluna 'userId'
topUsersRating = pd.merge(topUsers, ratings_df, on='userId', how='inner')

print(topUsersRating.head())

#Now all we need to do is simply multiply the movie rating by its weight (the similarity index), then sum up the new ratings and divide it by the sum of the weights.
#We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:

print("Rating of selected users to all movies : ")

#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
print(topUsersRating.head())

print("Top Users agropados por userId : ")

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
print(tempTopUsersRating.head())

print("Recomendações totais : ")

#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
print(recommendation_df.head())

print("Recomendações : ")

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))

print("Recomendações Finais : ")

print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())])
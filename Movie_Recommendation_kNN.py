import numpy as np
import pandas as pd
import operator


df_movies = pd.read_csv("movies.csv", usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
df_ratings = pd.read_csv("ratings_train.csv", usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
df_movie_features = df_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)



# Similarity and Distance functions
###################################

def cosine_similarity (a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def euclidean_distance(a,b):
    return np.linalg.norm(np.array(b)-np.array(a))



#Split the data set into train & validation
###########################################

train_dataset = df_movie_features.sample(frac=0.8,random_state=200)
validation_dataset = df_movie_features.drop(train_dataset.index)



#Calculate similarities between each user and create a user x user matrix
#########################################################################

def find_similarities():
	user_count = df_movie_features.shape[1]
	similarity_matrix = np.zeros((user_count, user_count))
	for i in range(train_dataset.shape[1]):
		for j in range(train_dataset.shape[1]):
			sim = cosine_similarity(train_dataset.values[i], train_dataset.values[j])
			similarity_matrix[i][j] = sim
	np.fill_diagonal(similarity_matrix, 0) #fill the diagonal with 0s so that the same users dont have 1.0 similarity
	return similarity_matrix



#Convert the user similarity matrix into an indexed dataframe
#############################################################

user_similarity = pd.DataFrame(find_similarities(), index= train_dataset.columns )
user_similarity.columns = train_dataset.columns



def knn(test_instance, n):
	neighbors = []
	test_instance = test_instance.sort_values(axis = 0, ascending = False)
	for x in range(n):
		neighbors.append(test_instance.index[x])
	return neighbors

def weighted_knn(test_instance, userID,  n):
	neighbors = []
	distances = []
	for i in range(user_similarity.shape[0]):
		if not(user_similarity.index[i] == userID):
			dist = euclidean_distance(user_similarity.iloc[i], test_instance)
			distances.append((user_similarity.index[i], dist))
	distances.sort(key=operator.itemgetter(1))
	for x in range(n):
		neighbors.append(distances[x][0])
	return neighbors


#Predicts rating of a given movie of a specific user.
#Rating is predicted as summing all the existing ratings of the same movie of the neighbors
# and taking the average.
##############################################################

def predict_rating(test_instance, movieID, userID, n, weighted):
	rating = 0

	if weighted:
		n_neighbors = weighted_knn(test_instance, userID, n)
	else:
		n_neighbors = knn(test_instance, n)
	count = 0

	for i in range(n):
		if not(df_movie_features.loc[movieID, n_neighbors[i]] == 0):
			rating += df_movie_features.loc[movieID, n_neighbors[i]]
			count += 1
	if  count == 0:
		return -1
	else:
		return rating / count

#Calculates Mean Absolute Error for the every rated movie in the validation dataset.
#However if none of the k nearest neighbors of the user has rated that movie, it's discarded.
#############################################################################################

def calculate_mae(n, weighted):
	errors_sum = 0
	count = 0
	for i in range(validation_dataset.shape[0]):
		for j in range(validation_dataset.shape[1]):
			if not (validation_dataset.iloc[i, j] == 0):
				userID = validation_dataset.columns[j]
				movieID = validation_dataset.index[i]
				predicted_rating = predict_rating(user_similarity.loc[userID,:], movieID, userID, n, weighted)
				if not(predicted_rating == -1):
					errors_sum += abs(predicted_rating - validation_dataset.iloc[i, j])
					count += 1
	return errors_sum / count

print(calculate_mae(10, True))

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from scipy.sparse import csr_matrix
# warnings.filterwarnings('ignore')



movies_df = pd.read_csv('movies.dat', sep='::', engine='python', 
                        encoding="ISO-8859-1", header=None,
                        names=['MovieID', 'Title', 'Genres'])

ratings_df = pd.read_csv('ratings.dat', sep='::', engine='python', 
                         header=None,
                         names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

ratings_matrix = pd.read_csv('Rmat.csv', index_col=0)


def get_popular_movies(ratings_df, movies_df, k=10, min_ratings=100):

    try:
        return pd.read_csv(f'popular_movies_{k}')
    except FileNotFoundError:
        # Calculate the number of ratings and average rating for each movie
        movie_stats = ratings_df.groupby('MovieID').agg({
            'Rating': ['count', 'mean']
        }).reset_index()
        
        movie_stats.columns = ['MovieID', 'rating_count', 'rating_mean']
        
        # Filter movies with minimum number of ratings
        qualified_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
        
        qualified_movies['popularity_score'] = (qualified_movies['rating_count'] * 
                                            qualified_movies['rating_mean'])
        
        top_movies = qualified_movies.nlargest(10, 'popularity_score')
        
        result = top_movies.merge(movies_df, on='MovieID')
        
        result['MovieID_prefix'] = 'm' + result['MovieID'].astype(str)
        
        return result[['MovieID_prefix', 'Title', 'rating_count', 
                    'rating_mean', 'popularity_score']]

top_popular_movies = get_popular_movies(ratings_df, movies_df)
print("Top 10 Most Popular Movies:")
print(top_popular_movies)


def normalize_ratings_matrix(ratings_matrix):
    """
    Normalize the rating matrix by centering each row (user).
    """
    row_means = ratings_matrix.mean(axis=1)
    
    normalized_matrix = ratings_matrix.sub(row_means, axis=0)
    
    return normalized_matrix

def compute_similarity_matrix(normalized_matrix):
    """
    Compute similarity between movies using the transformed cosine similarity.
    """
    movie_ids = normalized_matrix.columns
    n_movies = len(movie_ids)
    
    similarity_matrix = pd.DataFrame(
        np.nan, 
        index=movie_ids,
        columns=movie_ids
    )
    
    for i in range(n_movies):
        for j in range(i+1, n_movies):
            movie_i = movie_ids[i]
            movie_j = movie_ids[j]
            
            ratings_i = normalized_matrix[movie_i]
            ratings_j = normalized_matrix[movie_j]
            
            # Find common users
            common_mask = ~(ratings_i.isna() | ratings_j.isna())
            common_i = ratings_i[common_mask]
            common_j = ratings_j[common_mask]
            
            # Check if we have enough common ratings
            if len(common_i) <= 2:
                continue
            
            numerator = (common_i * common_j).sum()
            denom_i = np.sqrt((common_i**2).sum())
            denom_j = np.sqrt((common_j**2).sum())
            
            # Check for zero denominators
            if denom_i == 0 or denom_j == 0:
                continue
            
            # Calculate transformed cosine similarity
            cos_sim = numerator / (denom_i * denom_j)
            similarity = 0.5 + 0.5 * cos_sim
            
            similarity_matrix.loc[movie_i, movie_j] = similarity
            similarity_matrix.loc[movie_j, movie_i] = similarity
    
    return similarity_matrix


# In[ ]:


def keep_top_k_similarities(similarity_matrix, k=30):
    """
    For each row, keep only the top k highest similarities, setting rest to NA.
    Uses direct indexing rather than threshold approach.
    """
    result = similarity_matrix.copy()
    
    for movie in result.index:
        row = result.loc[movie].copy()
        
        top_k_indices = row.nlargest(k).index
        
        result.loc[movie, :] = np.nan
        result.loc[movie, top_k_indices] = row[top_k_indices]
    
    return result


# In[112]:


def myIBCF(newuser, similarity_matrix, popularity_rankings=None):
    """
    Input:
        newuser: 3706x1 vector with ratings (1-5) and NA values
        similarity_matrix: movie similarity matrix (top 30 version)
        popularity_rankings: backup rankings for when we have < 10 predictions
    
    Output:
        List of top 10 recommended movie IDs (with 'm' prefix)
    """
    rated_movies = newuser[~newuser.isna()].index
    predictions = {}
    
    for movie in similarity_matrix.index:
        # Skip if movie is already rated
        if movie in rated_movies:
            continue
        
        # Get similar movies
        similar_rated = similarity_matrix.loc[movie][rated_movies].dropna()
        
        # Skip if no similar movies were rated
        if len(similar_rated) == 0:
            continue
        
        user_ratings = newuser[similar_rated.index]
        
        weights_sum = similar_rated.sum()
        if weights_sum == 0:
            continue
            
        weighted_rating = (similar_rated * user_ratings).sum() / weights_sum
        predictions[movie] = weighted_rating
    
    # Convert predictions to Series for easier handling
    pred_series = pd.Series(predictions)
    
    # If we have fewer than 10 predictions, use popularity rankings
    if len(pred_series) < 10:
        if popularity_rankings is None:
            popularity_rankings = get_popular_movies(ratings_df, movies_df)['MovieID_prefix'].values
        
        top_pred = pred_series.nlargest(min(10, len(pred_series)))
        n_needed = 10 - len(top_pred)
        
        # Get popular movies not already rated or predicted
        backup_movies = [
            m for m in popularity_rankings 
            if m not in rated_movies and m not in top_pred.index
        ][:n_needed]
        
        recommendations = list(top_pred.index) + backup_movies
    else:
        recommendations = list(pred_series.nlargest(10).index)
    
    return recommendations[:10]



print("Normalizing ratings matrix...")
normalized_matrix = normalize_ratings_matrix(ratings_matrix)


# Read similarity matrix or compute it
try:
    similarity_matrix = pd.read_csv('similarity_matrix.csv')
    similarity_matrix_top30 = pd.read_csv('similarity_matrix_top30.csv')
except FileNotFoundError:
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(normalized_matrix)
    similarity_matrix.to_csv('similarity_matrix.csv')

    print("Keeping top 30 similarities per movie...")
    similarity_matrix_top30 = keep_top_k_similarities(similarity_matrix)
    similarity_matrix_top30.to_csv('similarity_matrix_top30.csv')

specified_movies = ['m1', 'm10', 'm100', 'm1510', 'm260', 'm3212']
similarity_subset = similarity_matrix.loc[specified_movies, specified_movies]
print("\nPairwise Similarity Values (from Step 2, rounded to 7 decimal places):")
print(similarity_subset.round(7))


# Test the function:
print("\nTest Case 1:")
u1181_ratings = ratings_matrix.loc['u1181']
recommendations_1 = myIBCF(u1181_ratings, similarity_matrix_top30)
print("Recommendations for u1181:", recommendations_1)

print("\nTest Case 2:")
hypothetical_user = pd.Series(np.nan, index=ratings_matrix.columns)
hypothetical_user['m1613'] = 5
hypothetical_user['m1755'] = 4
recommendations_2 = myIBCF(hypothetical_user, similarity_matrix_top30)
print("Recommendations for hypothetical user:", recommendations_2)





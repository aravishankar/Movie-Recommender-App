import pandas as pd
import numpy as np


def get_popular_movies(ratings_df, movies_df, k=10, min_ratings=100):
    try:
        return pd.read_csv(f'popular_movies_{k}')
    except FileNotFoundError:
        movie_stats = ratings_df.groupby('MovieID').agg({
            'Rating': ['count', 'mean']
        }).reset_index()
        
        movie_stats.columns = ['MovieID', 'rating_count', 'rating_mean']
        
        qualified_movies = movie_stats[movie_stats['rating_count'] >= min_ratings]
        
        qualified_movies['popularity_score'] = (qualified_movies['rating_count'] * 
                                            qualified_movies['rating_mean'])
        
        top_movies = qualified_movies.nlargest(10, 'popularity_score')
        
        result = top_movies.merge(movies_df, on='MovieID')
        
        result['MovieID_prefix'] = 'm' + result['MovieID'].astype(str)
        
        return result[['MovieID_prefix', 'Title', 'rating_count', 
                    'rating_mean', 'popularity_score']]


def normalize_ratings_matrix(ratings_matrix):
    row_means = ratings_matrix.mean(axis=1)
    
    normalized_matrix = ratings_matrix.sub(row_means, axis=0)
    
    return normalized_matrix


def compute_similarity_matrix(normalized_matrix):
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
            
            common_mask = ~(ratings_i.isna() | ratings_j.isna())
            common_i = ratings_i[common_mask]
            common_j = ratings_j[common_mask]
            
            if len(common_i) <= 2:
                continue
            
            numerator = (common_i * common_j).sum()
            denom_i = np.sqrt((common_i**2).sum())
            denom_j = np.sqrt((common_j**2).sum())
            
            if denom_i == 0 or denom_j == 0:
                continue
            
            cos_sim = numerator / (denom_i * denom_j)
            similarity = 0.5 + 0.5 * cos_sim
            
            similarity_matrix.loc[movie_i, movie_j] = similarity
            similarity_matrix.loc[movie_j, movie_i] = similarity
    
    return similarity_matrix


def keep_top_k_similarities(similarity_matrix, k=30):
    result = similarity_matrix.copy()
    
    for movie in result.index:
        row = result.loc[movie].copy()
        
        top_k_indices = row.nlargest(k).index
        
        result.loc[movie, :] = np.nan
        result.loc[movie, top_k_indices] = row[top_k_indices]
    
    return result


def myIBCF(newuser, similarity_matrix, ratings_df, movies_df, popularity_rankings=None):
    rated_movies = newuser[~newuser.isna()].index
    predictions = {}
    
    for movie in similarity_matrix.index:
        if movie in rated_movies:
            continue
        
        similar_rated = similarity_matrix.loc[movie][rated_movies].dropna()
        
        if len(similar_rated) == 0:
            continue
        
        user_ratings = newuser[similar_rated.index]
        
        weights_sum = similar_rated.sum()
        if weights_sum == 0:
            continue
            
        weighted_rating = (similar_rated * user_ratings).sum() / weights_sum
        predictions[movie] = weighted_rating
    
    pred_series = pd.Series(predictions)
    
    if len(pred_series) < 10:
        if popularity_rankings is None:
            popularity_rankings = get_popular_movies(ratings_df, movies_df)['MovieID_prefix'].values
        
        top_pred = pred_series.nlargest(min(10, len(pred_series)))
        n_needed = 10 - len(top_pred)
        
        backup_movies = [
            m for m in popularity_rankings 
            if m not in rated_movies and m not in top_pred.index
        ][:n_needed]
        
        recommendations = list(top_pred.index) + backup_movies
    else:
        recommendations = list(pred_series.nlargest(10).index)
    
    return recommendations[:10]
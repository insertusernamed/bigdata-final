import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time


def compute_movie_similarity(matrix):
    """calculates cosine similarity between movies using input rating matrix"""
    try:
        print("computing movie similarity matrix...")
        start_time = time.time()
        similarity_matrix = cosine_similarity(matrix.T)
        print(f"similarity matrix computed in {time.time() - start_time:.2f} seconds")
        print(f"similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix
    except Exception as e:
        print(f"error computing similarity matrix: {e}")
        return None


def save_similarity_matrix(similarity_matrix, base_path="./dataset"):
    """saves computed similarity matrix to compressed npz file"""
    try:
        similarity_matrix_path = f"{base_path}/movie_similarity_matrix.npz"
        np.savez_compressed(similarity_matrix_path, similarity_matrix)
        print(f"similarity matrix saved to {similarity_matrix_path}")
        return True
    except Exception as e:
        print(f"error saving similarity matrix: {e}")
        return False

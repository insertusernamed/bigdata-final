import numpy as np
from scipy.sparse import csr_matrix, save_npz
import pickle
import time


def create_matrix(ratings_df):
    """creates a sparse user-item matrix from a dataframe containing ratings
    returns the matrix and mappings of user/item ids to matrix indices"""
    try:
        print("creating user-item matrix...")
        print(f"number of unique users: {ratings_df['user_id'].nunique()}")
        print(f"number of unique items: {ratings_df['item_id'].nunique()}")
        print(f"total ratings: {len(ratings_df)}")
        start_time = time.time()
        user_ids = ratings_df["user_id"].unique()
        item_ids = ratings_df["item_id"].unique()
        user_map = {id: i for i, id in enumerate(user_ids)}
        item_map = {id: i for i, id in enumerate(item_ids)}
        row = ratings_df["user_id"].map(user_map)
        col = ratings_df["item_id"].map(item_map)
        data = ratings_df["rating"]
        matrix = csr_matrix(
            (data, (row, col)), shape=(len(user_ids), len(item_ids)), dtype=np.float32
        )
        print(f"matrix density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2%}")
        print(f"matrix created in {time.time() - start_time:.2f} seconds")
        print(f"matrix shape: {matrix.shape}")
        print(f"memory usage: {matrix.data.nbytes / 1024**2:.2f} mb")
        return matrix, user_map, item_map
    except Exception as e:
        print(f"error creating matrix: {e}")
        return None, None, None


def save_matrix_and_mappings(matrix, user_map, item_map, base_path="./dataset"):
    """saves the sparse matrix and id mappings to disk in npz and pickle formats"""
    try:
        matrix_path = f"{base_path}/user_item_matrix.npz"
        save_npz(matrix_path, matrix)
        with open(f"{base_path}/user_map.pkl", "wb") as f:
            pickle.dump(user_map, f)
        with open(f"{base_path}/item_map.pkl", "wb") as f:
            pickle.dump(item_map, f)
        print(f"matrix and mappings saved to {base_path}")
        return True
    except Exception as e:
        print(f"error saving files: {e}")
        return False

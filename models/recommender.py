import numpy as np
from scipy.sparse import load_npz
import pickle
import os
import pandas as pd


class MovieRecommender:
    """movie recommendation system using similarity matrix and movie metadata"""

    def __init__(self, base_path="./dataset"):
        """initializes recommender with model files and metadata from the specified path"""
        self.base_path = base_path
        self.similarity_matrix = None
        self.item_map = None
        self.reverse_item_map = None
        self.is_initialized = False
        self.metadata = None
        if not self.load_models():
            print(f"warning: models not loaded from {base_path}")
            print(f"please generate the dataset first using the generate buttons")
        self.load_metadata()

    def load_models(self):
        """loads similarity matrix and item mapping from saved files"""
        try:
            similarity_path = f"{self.base_path}/movie_similarity_matrix.npz"
            item_map_path = f"{self.base_path}/item_map.pkl"

            if not (os.path.exists(similarity_path) and os.path.exists(item_map_path)):
                print(f"required files not found in {self.base_path}")
                return False

            with np.load(similarity_path) as data:
                self.similarity_matrix = data["arr_0"]

            with open(item_map_path, "rb") as f:
                self.item_map = pickle.load(f)

            self.reverse_item_map = {v: k for k, v in self.item_map.items()}
            self.is_initialized = True
            print(f"models successfully loaded from {self.base_path}")
            return True
        except Exception as e:
            print(f"error loading models: {e}")
            return False

    def load_metadata(self):
        """loads movie metadata from json file"""
        try:
            self.metadata = pd.read_json(f"{self.base_path}/metadata.json", lines=True)
            self.metadata.set_index("item_id", inplace=True)
            print(f"metadata loaded successfully: {len(self.metadata)} movies")
        except Exception as e:
            print(f"error loading metadata: {e}")

    def get_movie_details(self, item_id):
        """retrieves detailed information for a specific movie by its id"""
        try:
            return self.metadata.loc[item_id].to_dict()
        except:
            return None

    def search_movies(self, query, limit=10):
        """searches for movies by title and returns matching results"""
        try:
            matches = self.metadata[
                self.metadata["title"].str.contains(query, case=False)
            ]
            matches = matches.reset_index()
            return matches.head(limit).to_dict("records")
        except Exception as e:
            print(f"error searching movies: {e}")
            return []

    def get_similar_movies(self, item_id, n=5):
        """finds n most similar movies to a given movie using similarity matrix"""
        if not self.is_initialized:
            print("recommender not initialized. please generate dataset first")
            return []

        try:
            idx = self.item_map.get(item_id)
            if idx is None:
                return []

            similar_scores = self.similarity_matrix[idx]
            similar_movies = np.argsort(similar_scores)[-n - 1 :][::-1]
            similar_movies = similar_movies[similar_movies != idx]

            recommendations = []
            for movie_idx in similar_movies[:n]:
                movie_id = int(self.reverse_item_map[int(movie_idx)])
                movie_info = self.get_movie_details(movie_id)
                if movie_info:
                    movie_info["similarity_score"] = float(similar_scores[movie_idx])
                    recommendations.append(movie_info)

            return recommendations
        except Exception as e:
            print(f"error getting similar movies: {e}")
            return []

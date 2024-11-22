from flask import Flask, jsonify, request, render_template
from utils.data_loader import load_sample_data, load_full_data, load_metadata
from utils.matrix_ops import create_matrix, save_matrix_and_mappings
from models.similarity import compute_movie_similarity, save_similarity_matrix
from models.recommender import MovieRecommender

app = Flask(__name__)
recommender = MovieRecommender(base_path="./dataset")


@app.route("/")
def index():
    """serve the main page of the application"""
    return render_template("index.html")


@app.route("/load_sample", methods=["GET"])
def load_sample():
    """load and process a sample of the ratings data"""
    ratings_sample = load_sample_data("./dataset/ratings.json", sample_size=100000)
    if ratings_sample is not None:
        matrix, user_map, item_map = create_matrix(ratings_sample)
        if matrix is not None:
            save_matrix_and_mappings(
                matrix, user_map, item_map, base_path="./dataset/sample"
            )
            similarity_matrix = compute_movie_similarity(matrix)
            if similarity_matrix is not None:
                save_similarity_matrix(similarity_matrix, base_path="./dataset/sample")
            return jsonify({"message": "sample data processed successfully"}), 200
    return jsonify({"message": "failed to process sample data"}), 500


@app.route("/recommend/similar/<int:item_id>")
def get_similar_movies(item_id):
    """get similar movie recommendations for a given movie id"""
    n = request.args.get("n", default=5, type=int)
    recommendations = recommender.get_similar_movies(item_id, n=n)
    if recommendations:
        return jsonify({"item_id": item_id, "recommendations": recommendations}), 200
    return jsonify({"message": "could not get recommendations"}), 404


@app.route("/generate/sample", methods=["POST"])
def generate_sample():
    """generate and process a sample dataset"""
    ratings_sample = load_sample_data("./dataset/ratings.json", sample_size=100000)
    return process_data(ratings_sample, base_path="./dataset/sample")


@app.route("/generate/full", methods=["POST"])
def generate_full():
    """generate and process the full dataset"""
    ratings_full = load_full_data("./dataset/ratings.json")
    return process_data(ratings_full, base_path="./dataset")


@app.route("/search")
def search_movies():
    """search for movies based on a query string"""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"message": "no search query provided"}), 400
    results = recommender.search_movies(query)
    return jsonify({"results": results})


@app.route("/movie/<int:item_id>")
def get_movie_details(item_id):
    """get detailed information about a specific movie"""
    details = recommender.get_movie_details(item_id)
    if details:
        return jsonify(details)
    return jsonify({"message": "movie not found"}), 404


def process_data(ratings_data, base_path):
    """process ratings data and generate similarity matrices"""
    if ratings_data is not None:
        matrix, user_map, item_map = create_matrix(ratings_data)
        if matrix is not None:
            save_matrix_and_mappings(matrix, user_map, item_map, base_path=base_path)
            similarity_matrix = compute_movie_similarity(matrix)
            if similarity_matrix is not None:
                save_similarity_matrix(similarity_matrix, base_path=base_path)
                global recommender
                recommender = MovieRecommender(base_path=base_path)
                return jsonify({"message": "data processed successfully"}), 200
    return jsonify({"message": "failed to process data"}), 500


if __name__ == "__main__":
    app.run(debug=True)

# Movie Recommender System: Final Project in Big Data

## Overview

This project is a movie recommender system built using the MovieLens 1.8GB Tag Genome dataset. The system leverages machine learning algorithms and Flask for serving recommendations in a web interface. It provides movie recommendations based on user preferences and ratings, using various features such as movie tags, metadata, and user ratings.

## Dataset

The project uses the **MovieLens 1.8GB Tag Genome dataset**. This dataset includes movie metadata, user ratings, and movie tags. The key files used for this project are:

- `ratings.csv` — Contains user ratings for movies.
- `metadata.csv` — Contains metadata about the movies (e.g., genres, titles).
- `tag_genome.csv` — Contains movie tags, which are used for filtering and recommendations.

You can download the dataset from [MovieLens Tag Genome Dataset (genome_2021.zip)](https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip).

## Requirements

To run this project locally, you need to have Python 3.x installed, along with a few necessary libraries.

### Install Python Libraries

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/movie-recommender.git
    cd movie-recommender
    ```

2. Download the dataset from [MovieLens Tag Genome Dataset](https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip), extract it, and place the following files into the `datasets/` folder in your project:
    - `ratings.csv`
    - `metadata.csv`

3. Install the required Python libraries:
    ```bash
    pip install scikit-learn flask
    ```

4. Run the app:
    ```bash
    python app.py
    ```

# Movie Recommendation Engine


A machine learning-based recommendation system using the MovieLens 100K dataset and KMDB 5000 dataset implementing multiple recommendation approaches.

## Overview

This project implements three different recommendation algorithms:
- Collaborative Filtering using K-Nearest Neighbors
- Softmax Decomposition-Discovery (DD)
- Content-Based Filtering using Cosine Similarity

## Dataset

MovieLens 100K dataset containing:
- 100,000 ratings (1-5) from 943 users on 1,682 movies
- Movie metadata including title, genre, release date
- User demographic information

KMDB 5000 dataset containing:
- 4804 movies with 23 input features inlcuding overview, genres, cast, crew, ratings

## Implementation Details

### Collaborative Filtering (KNN)
- User-based approach using k-nearest neighbors
- K-value optimized through cross-validation

### Softmax DD
- Learns latent factors for users and items
- Incorporates user and item biases

### Collaborative Filtering(SVD)
- item-based approach using SVD
- Trains the model, generates predictions, and evaluates with RMSE and MAE metrics
- Requires a dataset located at data/ml-100k to run and evaluate the model.

### Content-Based Filtering
- Feature extraction from movie metadata
- TF-IDF vectorization for movie descriptions
- Cosine similarity for recommendations
- Jaccard Similarity for model validation using internet data



## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Jaccard Similarity



# Setting Up Virtual Environments

## Using `venv` (Python's Built-In Virtual Environment)

1. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   ```

2. Activate the virtual environment:
   - On Unix/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

---

## Using `conda`

1. Create a conda environment from an `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the conda environment:
   ```bash
   conda activate movie-recommender
   ```

---

## Using `pyenv`

1. Install a specific version of Python with `pyenv`:
   ```bash
   pyenv install 3.11.2
   ```

2. Set the local Python version:
   ```bash
   pyenv local 3.11.2
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

---

## Alternative: Using a `make` Command

If `pyenv` is installed on your system, you can also set up your environment using a `make` command:

1. Run the following command to set up the environment:
   ```bash
   make setup
   ```

---


## Dataset and Implementation Details Of NLP Methord

This project implements three different recommendation algorithms:
- Dataset Used: TMDB 5000 Movies Dataset.
- Embedding with Universal Sentence Encoder: Movie overviews are transformed into embeddings using the Universal Sentence Encoder model.
- Dimensionality Reduction with PCA: The high-dimensional embeddings are reduced to 2D for better visualization.
- Visualization: The reduced embedding space is plotted using Matplotlib to provide an intuitive view.
- Recommendation Implementation: The Nearest Neighbors algorithm is employed to recommend similar movies based on the generated embeddings.


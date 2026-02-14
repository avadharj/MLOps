import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import logging
import os
import json

logger = logging.getLogger(__name__)

# ============================================================
# CHANGE 1: New dataset - Iris dataset from sklearn
# Original lab used a custom CSV file (file.csv)
# ============================================================

def load_data():
    """
    Loads the Iris dataset from sklearn instead of the original CSV file.
    This is a well-known dataset with 4 features (sepal length, sepal width,
    petal length, petal width) and 150 samples across 3 species.
    
    Returns:
        serialized pandas DataFrame
    """
    logger.info("Loading Iris dataset from sklearn...")
    
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    # Add the target column for later reference (not used in clustering)
    df['species'] = iris.target
    df['species_name'] = [iris.target_names[i] for i in iris.target]
    
    logger.info(f"Loaded Iris dataset with shape: {df.shape}")
    logger.info(f"Features: {iris.feature_names}")
    logger.info(f"Sample data:\n{df.head()}")
    
    serialized_data = pickle.dumps(df)
    return serialized_data


def data_preprocessing(data):
    """
    Deserializes data, performs preprocessing including:
    - Dropping non-numeric columns (species labels)
    - StandardScaler normalization (new addition)
    
    Returns:
        serialized preprocessed DataFrame
    """
    logger.info("Starting data preprocessing...")
    
    df = pickle.loads(data)
    
    # Keep only numeric feature columns for clustering
    feature_cols = [col for col in df.columns if col not in ['species', 'species_name']]
    df_features = df[feature_cols].copy()
    
    # Drop any rows with missing values
    df_features = df_features.dropna()
    
    # Standardize features (important for distance-based clustering)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_data, columns=feature_cols)
    
    logger.info(f"Preprocessed data shape: {df_scaled.shape}")
    logger.info(f"Scaled feature statistics:\n{df_scaled.describe()}")
    
    serialized_data = pickle.dumps(df_scaled)
    return serialized_data


def build_save_model(data, filename):
    """
    Builds KMeans clustering models for k=2..10, saves the best model,
    and returns SSE values for elbow method analysis.
    
    Args:
        data: serialized preprocessed DataFrame
        filename: name of the file to save the model
    
    Returns:
        serialized list of SSE values
    """
    logger.info("Building KMeans clustering models...")
    
    df = pickle.loads(data)
    
    sse = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        logger.info(f"KMeans k={k}: SSE = {kmeans.inertia_:.2f}")
    
    # Save the model with k=3 (since Iris has 3 species) as default
    best_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    best_kmeans.fit(df)
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_kmeans, f)
    
    logger.info(f"KMeans model saved to {model_path}")
    
    serialized_sse = pickle.dumps(sse)
    return serialized_sse


def load_model_elbow(filename, sse):
    """
    Loads the saved KMeans model and determines optimal clusters
    using the elbow method (KneeLocator).
    
    Args:
        filename: model filename
        sse: serialized SSE values
    
    Returns:
        optimal number of clusters
    """
    logger.info("Determining optimal clusters using elbow method...")
    
    sse_values = pickle.loads(sse)
    k_range = list(range(2, 11))
    
    kl = KneeLocator(
        k_range, sse_values,
        curve="convex",
        direction="decreasing"
    )
    
    optimal_k = kl.elbow
    logger.info(f"Elbow Method - Optimal number of clusters: {optimal_k}")
    
    # Load and verify the saved model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    model_path = os.path.join(model_dir, filename)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded KMeans model with {model.n_clusters} clusters")
    
    return optimal_k


# ============================================================
# CHANGE 2: New DBSCAN clustering task
# Original lab only used KMeans
# ============================================================

def build_dbscan_model(data):
    """
    Builds a DBSCAN clustering model as an alternative to KMeans.
    DBSCAN can find arbitrarily shaped clusters and automatically
    determines the number of clusters.
    
    Args:
        data: serialized preprocessed DataFrame
    
    Returns:
        serialized dict with DBSCAN results
    """
    logger.info("Building DBSCAN clustering model...")
    
    df = pickle.loads(data)
    
    # Try different eps values to find good clustering
    best_score = -1
    best_eps = 0.5
    best_labels = None
    
    for eps in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(df)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters >= 2:
            score = silhouette_score(df, labels)
            logger.info(f"DBSCAN eps={eps}: {n_clusters} clusters, "
                       f"silhouette={score:.4f}, noise={list(labels).count(-1)} points")
            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = labels
    
    if best_labels is None:
        logger.warning("DBSCAN could not find valid clusters. Using default eps=0.5")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        best_labels = dbscan.fit_predict(df)
    
    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise = list(best_labels).count(-1)
    
    results = {
        'n_clusters': n_clusters,
        'n_noise_points': n_noise,
        'best_eps': best_eps,
        'best_silhouette': best_score,
        'labels': best_labels.tolist()
    }
    
    # Save DBSCAN model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    dbscan_path = os.path.join(model_dir, 'dbscan_model.sav')
    
    final_dbscan = DBSCAN(eps=best_eps, min_samples=5)
    final_dbscan.fit(df)
    
    with open(dbscan_path, 'wb') as f:
        pickle.dump(final_dbscan, f)
    
    logger.info(f"DBSCAN Results: {n_clusters} clusters found, "
               f"{n_noise} noise points, best eps={best_eps}")
    
    serialized_results = pickle.dumps(results)
    return serialized_results


# ============================================================
# CHANGE 3: Silhouette score evaluation task
# Original lab only used the elbow method
# ============================================================

def evaluate_models(data, dbscan_results):
    """
    Evaluates and compares KMeans and DBSCAN models using
    silhouette scores. Produces a summary comparison.
    
    Args:
        data: serialized preprocessed DataFrame
        dbscan_results: serialized DBSCAN results dict
    
    Returns:
        comparison summary string
    """
    logger.info("Evaluating and comparing clustering models...")
    
    df = pickle.loads(data)
    dbscan_res = pickle.loads(dbscan_results)
    
    # Evaluate KMeans for different k values
    kmeans_scores = {}
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        score = silhouette_score(df, labels)
        kmeans_scores[k] = score
        logger.info(f"KMeans k={k}: Silhouette Score = {score:.4f}")
    
    best_k = max(kmeans_scores, key=kmeans_scores.get)
    best_kmeans_score = kmeans_scores[best_k]
    
    # DBSCAN score
    dbscan_score = dbscan_res['best_silhouette']
    dbscan_clusters = dbscan_res['n_clusters']
    
    # Summary
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset: Iris (150 samples, 4 features)")
    logger.info(f"")
    logger.info(f"KMeans Results:")
    for k, score in kmeans_scores.items():
        marker = " <-- BEST" if k == best_k else ""
        logger.info(f"  k={k}: Silhouette = {score:.4f}{marker}")
    logger.info(f"")
    logger.info(f"DBSCAN Results:")
    logger.info(f"  Clusters found: {dbscan_clusters}")
    logger.info(f"  Noise points: {dbscan_res['n_noise_points']}")
    logger.info(f"  Best eps: {dbscan_res['best_eps']}")
    logger.info(f"  Silhouette Score: {dbscan_score:.4f}")
    logger.info(f"")
    
    if best_kmeans_score > dbscan_score:
        winner = f"KMeans (k={best_k}) with score {best_kmeans_score:.4f}"
    else:
        winner = f"DBSCAN (eps={dbscan_res['best_eps']}) with score {dbscan_score:.4f}"
    
    logger.info(f"WINNER: {winner}")
    logger.info("=" * 60)
    
    # Save comparison results to JSON
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    results_path = os.path.join(results_dir, 'comparison_results.json')
    
    comparison = {
        'dataset': 'Iris (sklearn)',
        'n_samples': len(df),
        'n_features': len(df.columns),
        'kmeans_scores': {str(k): round(v, 4) for k, v in kmeans_scores.items()},
        'kmeans_best_k': best_k,
        'kmeans_best_score': round(best_kmeans_score, 4),
        'dbscan_clusters': dbscan_clusters,
        'dbscan_noise_points': dbscan_res['n_noise_points'],
        'dbscan_eps': dbscan_res['best_eps'],
        'dbscan_score': round(dbscan_score, 4),
        'winner': winner
    }
    
    with open(results_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison results saved to {results_path}")
    
    return winner
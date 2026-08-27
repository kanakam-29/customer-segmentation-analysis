"""
Customer Segmentation Analysis
--------------------------------
Source code implementing customer segmentation using K-Means clustering.

Program Logic:
    1. Load and clean the customer dataset
    2. Select and scale relevant features
    3. Determine the optimal number of clusters (Elbow Method + Silhouette Score)
    4. Train the final K-Means model
    5. Assign cluster labels back to each customer
    6. Visualize the resulting segments
    7. Summarize/interpret each segment for business use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the customer dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the Mall_Customers.csv file.

    Returns
    -------
    pd.DataFrame
        Raw customer dataset.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and handle missing values and duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset (duplicates removed, no missing values).
    """
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"Missing values: {missing}, Duplicate rows: {duplicates}")

    df = df.drop_duplicates()
    df = df.dropna()
    return df


def select_and_scale_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Select clustering features and scale them using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    feature_cols : list
        List of column names to use for clustering.

    Returns
    -------
    tuple(np.ndarray, StandardScaler)
        Scaled feature matrix and the fitted scaler (for reuse on new data).
    """
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_k(X_scaled: np.ndarray, k_range=range(1, 11)) -> tuple:
    """
    Compute WCSS (inertia) for a range of k values (Elbow Method) and
    Silhouette Scores for k >= 2, to help determine the optimal number
    of clusters.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of k values to test.

    Returns
    -------
    tuple(list, dict)
        WCSS values per k, and silhouette scores per k (k >= 2).
    """
    wcss = []
    silhouette_scores = {}

    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        wcss.append(model.inertia_)

        if k >= 2:
            silhouette_scores[k] = silhouette_score(X_scaled, labels)

    return wcss, silhouette_scores


def plot_elbow_curve(k_range, wcss: list) -> None:
    """Plot the WCSS values against k to visualize the Elbow Method."""
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), wcss, marker="o", color="#2E5EAA")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.show()


def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """
    Train the final K-Means model with the chosen number of clusters.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k : int
        Number of clusters to use.

    Returns
    -------
    KMeans
        The trained K-Means model.
    """
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    model.fit(X_scaled)
    return model


def assign_clusters(df: pd.DataFrame, model: KMeans, X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Assign a cluster label to each customer record.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer dataset.
    model : KMeans
        Trained K-Means model.
    X_scaled : np.ndarray
        Scaled feature matrix used for prediction.

    Returns
    -------
    pd.DataFrame
        Dataset with an added 'Cluster' column.
    """
    df = df.copy()
    df["Cluster"] = model.predict(X_scaled)
    return df


def visualize_clusters(df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Plot the customer segments as a labeled scatter plot."""
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue="Cluster", palette="Set1", s=70)
    plt.title("Customer Segments (K-Means)")
    plt.show()


def summarize_segments(df: pd.DataFrame, profile_cols: list) -> pd.DataFrame:
    """
    Produce a per-cluster summary (mean values + customer count) to help
    interpret and label each segment.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with cluster labels assigned.
    profile_cols : list
        Numeric columns to average per cluster.

    Returns
    -------
    pd.DataFrame
        Summary table with mean feature values and count per cluster.
    """
    summary = df.groupby("Cluster")[profile_cols].mean().round(1)
    summary["Count"] = df["Cluster"].value_counts().sort_index()
    return summary


def main():
    """Run the full customer segmentation pipeline end-to-end."""
    # 1. Load and clean data
    df = load_data("Mall_Customers.csv")
    df = clean_data(df)

    # 2. Select and scale features
    feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled, scaler = select_and_scale_features(df, feature_cols)

    # 3. Determine optimal k
    wcss, sil_scores = find_optimal_k(X_scaled)
    plot_elbow_curve(range(1, 11), wcss)
    print("Silhouette scores by k:", sil_scores)

    # 4. Train final model (k=5 chosen from elbow + silhouette analysis)
    k_optimal = 5
    model = train_kmeans(X_scaled, k_optimal)

    # 5. Assign clusters
    df = assign_clusters(df, model, X_scaled)

    # 6. Visualize
    visualize_clusters(df, feature_cols[0], feature_cols[1])

    # 7. Interpret segments
    summary = summarize_segments(df, ["Age"] + feature_cols)
    print(summary)


if __name__ == "__main__":
    main()

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def print_dataset(df: pd.DataFrame, name: str):
    clusters = df["cluster"].unique()

    plt.figure(figsize=[12, 12])
    for cluster in clusters:
        points = df.loc[df["cluster"] == cluster]
        if cluster == -1:
            plt.scatter(points["x"], points["y"], marker="o", s=50, color="black")
        else:
            plt.scatter(points["x"], points["y"], marker="o", s=50)
    plt.title(f"Dataset: {name}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def print_data_with_outliers(data: np.array, outlier_scores: np.array, name: str):
    no_rows, no_cols = data.shape
    assert no_rows == outlier_scores.shape

    # Step 1. Plot the data and the outlier scores
    plt.figure(figsize=[12, 12])
    plt.scatter(data[:, 0], data[:, 1], s=25, c=outlier_scores, cmap="viridis")
    plt.colorbar()

    # Step 2: Assign rankings ~30% top outlier scores and plot top outliers
    top_30 = round(no_rows * 0.3)
    top_indices = np.argsort(outlier_scores)[-top_30:][::-1]  # Top 30 outliers, highest scores first
    for i, idx in enumerate(top_indices):
        plt.text(data[idx][0], data[idx][1], str(i + 1), fontsize=10, color="black")

    plt.title(name)
    plt.axis("equal")
    plt.show()

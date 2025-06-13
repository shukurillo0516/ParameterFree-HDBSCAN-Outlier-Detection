import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def print_dataset(dataset: pd.DataFrame):
    inliers_x = dataset.loc[dataset["labels"] == 0].iloc[:, 0]
    inliers_y = dataset.loc[dataset["labels"] == 0].iloc[:, 1]
    outliers_x = dataset.loc[dataset["labels"] == 1].iloc[:, 0]
    outliers_y = dataset.loc[dataset["labels"] == 1].iloc[:, 1]
    plt.scatter(inliers_x, inliers_y, marker="o", s=50)
    plt.scatter(outliers_x, outliers_y, color="red", marker="o", s=50)
    plt.title("Dataset")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def print_glosh_profiles(inlier_path: str, outlier_path: str, best_mpts: int | None = None):
    dataset_type = inlier_path.split("/")[-2]
    dataset_name = inlier_path.split("/")[-1].replace(".csv", "")

    labels = pd.read_csv(inlier_path)["labels"].to_numpy()
    df = pd.read_csv(outlier_path)  # Add header=None if needed
    outliers = df.to_numpy()

    rows_c, x = outliers.shape
    x_axis = [i + 2 for i in range(x)]

    plt.figure(figsize=(12, 8))
    for i in range(rows_c):
        y = outliers[i]
        if labels[i]:
            plt.plot(x_axis, y, linestyle="-", linewidth=0.5, color="red")
        else:
            plt.plot(x_axis, y, linestyle="-", linewidth=0.5, color="grey")

    if best_mpts:
        plt.axvline(x=best_mpts, color="green", linestyle="--", linewidth=2, label=f"m* = {best_mpts}")
    plt.title(f"{dataset_type}: {dataset_name}")
    plt.xlim(left=x_axis[0])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def plot_pn_scores(scores: list, path: str, best_mpts: int | None = None):
    dataset_type = path.split("/")[-2]
    dataset_name = path.split("/")[-1].replace(".csv", "")

    x_axis = [i + 2 for i in range(len(scores))]
    plt.figure(figsize=(12, 8))

    if best_mpts:
        plt.axvline(x=best_mpts, color="green", linestyle="--", linewidth=2, label=f"m* = {best_mpts}")
    plt.plot(x_axis, scores, linestyle="-", linewidth=0.5, color="blue")
    plt.title(f"{dataset_type}: {dataset_name}")
    plt.xlim(left=x_axis[0])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


def print_auto_glosh_scores(path: str, glosh_scores: np.array, dataset: pd.DataFrame | None = None):
    dataset_type = path.split("/")[-2]
    dataset_name = path.split("/")[-1].replace(".csv", "")

    if dataset is None or "labels" not in dataset.columns:
        raise ValueError("Dataset is required and must contain 'labels' column.")

    labels = dataset["labels"].to_numpy()

    # Step 1: Combine scores and labels
    combined = np.column_stack((glosh_scores, labels))

    # Step 2: Sort by scores
    sorted_combined = combined[np.argsort(combined[:, 0])]

    # Step 3: Add new indices after sorting (starting from 1)
    sorted_indices = np.arange(1, sorted_combined.shape[0] + 1).reshape(-1, 1)

    # Step 4: Combine sorted data with indices
    glosh_scores = np.hstack((sorted_combined, sorted_indices))

    print(glosh_scores)

    outliers = glosh_scores[glosh_scores[:, 1] == 1]
    inliers = glosh_scores[glosh_scores[:, 1] == 0]

    plt.figure(figsize=(12, 8))

    plt.scatter(outliers[:, 2], outliers[:, 0], marker="o", s=25, color="red")
    plt.scatter(inliers[:, 2], inliers[:, 0], marker="o", s=25)
    plt.title(f"{dataset_type}: {dataset_name}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

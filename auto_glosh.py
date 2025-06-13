import os
import subprocess
import csv
import pandas as pd
import numpy as np
import glob
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import hdbscan

from utils import find_max_index

from find_elbow.elbow import ElbowMptsFinder


def find_best_min_pts(profiles: pd.DataFrame):
    similarity = []
    for mpts in range(1, 100):
        similarity.append(1 - abs(stats.pearsonr(profiles.iloc[:, mpts - 1], profiles.iloc[:, mpts])[0]))

    similarity = np.array(similarity[:-2] if len(profiles) < 100 else similarity)
    arg = find_max_index(similarity)

    # Find elbow point
    elbowf = ElbowMptsFinder(range(len(similarity[arg:])), similarity[arg:])
    elbow_idx, _ = elbowf.find_elbow()
    mpts_star = int(elbow_idx[0]) + arg + 1
    return mpts_star + 2


def get_auto_glosh_scores(profiles: pd.DataFrame) -> np.array:
    best_mpts = find_best_min_pts(profiles)
    return profiles.iloc[:, best_mpts - 2]


# Auto-GLOSH algorithm
def auto_glosh(profiles, data):
    mpts_star = find_best_min_pts(profiles)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        algorithm="generic",
        alpha=1.0,
        approx_min_span_tree=False,
        gen_min_span_tree=True,
        metric="euclidean",
        min_cluster_size=mpts_star,
        min_samples=mpts_star,
    )
    clusterer.fit(data.iloc[:, :-1])
    glosh_scores = np.nan_to_num(clusterer.outlier_scores_, nan=0)
    return glosh_scores


if __name__ == "__main__":
    with open("metadata.csv", "r") as f:
        reader = csv.DictReader(f)
        results = []

        for row in reader:
            path = os.path.join(row["Path"], row["Name"])
            profiles_path = path.replace(".csv", "_outliers_python.csv")

            profiels = pd.read_csv(profiles_path)

            try:
                mpts_star = find_best_min_pts(profiels)
                results.append([path, mpts_star])
            except Exception as err:
                continue

    with open("output/best_mpts_python.csv", "w") as f:
        header = ["Path", "mpts_star"]
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

import subprocess
import pandas as pd
import numpy as np
import glob
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import hdbscan

from utils import find_max_index

from find_elbow.elbow import ElbowMptsFinder


def find_best_min_pts(profiles):
    similarity = []
    for mpts in range(1, 101):
        similarity.append(1 - abs(stats.pearsonr(profiles.iloc[:, mpts - 1], profiles.iloc[:, mpts])[0]))

    similarity = np.array(similarity[:-2] if len(profiles) < 100 else similarity)
    arg = find_max_index(similarity)

    # Find elbow point
    elbowf = ElbowMptsFinder(range(len(similarity[arg:])), similarity[arg:])
    elbow_idx, _ = elbowf.find_elbow()
    mpts_star = int(elbow_idx[0]) + arg + 1
    return mpts_star + 2


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
    result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
    print(result.stdout)  # Standard output
    print(result.stderr)  # Standard error (if any)
    print(result.returncode)  # Exit status

# profiles = pd.read_csv("datasets/real/Classical/4_breastw/4_breastw_test_glosh_scores.csv")
# data = pd.read_csv("datasets/real/Classical/4_breastw/4_breastw_test.csv")
# # print(data)
# # print("###############")
# # print(data.iloc[:, 2])
# auto_glosh(profiles, data)
# # for score in auto_glosh(profiles, data):
# #     print(score)

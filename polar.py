import pandas as pd
import numpy as np
import glob
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import hdbscan

from utils import find_max_index

# Auto-GLOSH algorithm
def auto_glosh(profiles, data):
    similarity = []
    for mpts in range(1, 101):
        similarity.append(1 - abs(stats.pearsonr(profiles.iloc[:, mpts - 1], profiles.iloc[:, mpts])[0]))

    similarity = np.array(similarity[:-2] if len(profiles) < 100 else similarity)
    arg = find_max_index(similarity)

    # Find knee point
    kneedle = KneeFinderSim(range(len(similarity[arg:])), similarity[arg:])
    knee, _ = kneedle.find_knee()
    mpts = int(knee[0]) + arg + 1

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        algorithm='generic', alpha=1.0, approx_min_span_tree=False,
        gen_min_span_tree=True, metric='euclidean',
        min_cluster_size=mpts + 2, min_samples=mpts + 2
    )
    clusterer.fit(data.iloc[:, :-1])
    glosh_scores = np.nan_to_num(clusterer.outlier_scores_, nan=0)
    return glosh_scores
import pandas as pd
import numpy as np
import glob
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import hdbscan

from utils import find_max_index

from find_elbow.elbow import ElbowMptsFinder

# Auto-GLOSH algorithm
def auto_glosh(profiles, data):
    similarity = []
    for mpts in range(1, 101):
        similarity.append(1 - abs(stats.pearsonr(profiles.iloc[:, mpts - 1], profiles.iloc[:, mpts])[0]))

    similarity = np.array(similarity[:-2] if len(profiles) < 100 else similarity)
    arg = find_max_index(similarity)

    # Find elbow point
    elbowf = ElbowMptsFinder(range(len(similarity[arg:])), similarity[arg:])
    elbow_idx, _ = elbowf.find_elbow()
    mpts_star = int(elbow_idx[0]) + arg + 1

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        algorithm='generic', alpha=1.0, approx_min_span_tree=False,
        gen_min_span_tree=True, metric='euclidean',
        min_cluster_size=mpts_star + 2, min_samples=mpts_star + 2
    )
    clusterer.fit(data.iloc[:, :-1])
    glosh_scores = np.nan_to_num(clusterer.outlier_scores_, nan=0)
    return glosh_scores
import numpy as np
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import hdbscan

from utils import find_max_index, line_through_points
from intersect import intersection

from find_knee.knee import KneeThersholdFinder
import pandas as pd


# POLAR algorithm
def polar(glosh_scores: np.array, data: pd.DataFrame):
    glosh_scores_lab = sorted(
        [[score, idx, label] for idx, (score, label) in enumerate(zip(glosh_scores, data.iloc[:, -1]))],
        key=lambda x: x[0],
    )
    sorted_glosh_scores = sorted(glosh_scores)

    # Find knee point
    indexes = list(range(len(sorted_glosh_scores)))
    kneeThres = KneeThersholdFinder(indexes, sorted_glosh_scores)
    knee_idx, _ = kneeThres.find_knee()
    knee_idx = int(knee_idx[0])

    # Intersection and thresholds
    intersections = kneeThres.find_intersection_points()
    # print(f"Length of sorted_glosh_scores: {len(sorted_glosh_scores)}")
    # print(f"Knee index: {knee_idx}")
    # print(f"Number of intersection points: {len(intersections)}")

    # intersect = intersections[knee_idx]
    safe_knee_idx = min(knee_idx, len(intersections) - 1)
    intersect = intersections[safe_knee_idx]
    # intersect = intersections[next(idx for idx, cntr in enumerate(indexes) if cntr == knee_idx)]

    x0, y0 = intersect[0][0], intersect[1][0]
    x2, y2 = (
        indexes[-1],
        linregress(indexes, sorted_glosh_scores)[0] * indexes[-1] + linregress(indexes, sorted_glosh_scores)[1],
    )

    x_line, y_line = line_through_points(x0, y0, x2, y2)
    x_intersects, _, _, _ = intersection(indexes, sorted_glosh_scores, x_line, y_line)
    knee_threshold = sorted_glosh_scores[knee_idx]
    adjusted_knee_threshold = sorted_glosh_scores[round(x_intersects[-1])]

    return knee_threshold, adjusted_knee_threshold

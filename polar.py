import numpy as np
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import hdbscan

from utils import find_max_index, line_through_points
from intersect import intersection

from find_knee.knee import KneeThersholdFinder


# POLAR algorithm
def polar(glosh_scores, data):
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
    intersections = kneeThres.find_intersection_point()
    intersect = intersections[next(idx for idx, cntr in enumerate(indexes) if cntr == knee_idx)]

    x0, y0 = intersect[0][0], intersect[1][0]
    x2, y2 = (
        indexes[-1],
        linregress(indexes, sorted_glosh_scores)[0] * indexes[-1] + linregress(indexes, sorted_glosh_scores)[1],
    )

    x_line, y_line = line_through_points(x0, y0, x2, y2)
    x_intersects, _ = intersection(indexes, sorted_glosh_scores, x_line, y_line)
    knee_threshold = sorted_glosh_scores[knee_idx]
    adjusted_knee_threshold = sorted_glosh_scores[round(x_intersects[-1])]

    return knee_threshold, adjusted_knee_threshold

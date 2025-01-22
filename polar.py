import pandas as pd
import numpy as np
import glob
from scipy.stats import linregress
from scipy import stats
import matplotlib.pyplot as plt
import hdbscan

from utils import find_max_index

# POLAR algorithm
def polar(glosh_scores, data):
    glosh_scores_lab = sorted([[score, idx, label] for idx, (score, label) in enumerate(zip(glosh_scores, data.iloc[:, -1]))], key=lambda x: x[0])
    sorted_glosh_scores = sorted(glosh_scores)

    # Find knee point
    indexes = list(range(len(sorted_glosh_scores)))
    kf = KneeFinder(indexes, sorted_glosh_scores)
    knee_x, _ = kf.find_knee()
    knee_x = int(knee_x[0])

    # Intersection and thresholds
    intersections = kf.find_intersection_point()
    intersect = intersections[next(idx for idx, cntr in enumerate(indexes) if cntr == knee_x)]

    x0, y0 = intersect[0][0], intersect[1][0]
    x2, y2 = indexes[-1], linregress(indexes, sorted_glosh_scores)[0] * indexes[-1] + linregress(indexes, sorted_glosh_scores)[1]

    x_line, y_line = line_through_points(x0, y0, x2, y2)
    x_intersects, _ = intersection(indexes, sorted_glosh_scores, x_line, y_line)
    inlier_thres_1 = sorted_glosh_scores[knee_x]
    inlier_thres_2 = sorted_glosh_scores[round(x_intersects[-1])]

    return inlier_thres_1, inlier_thres_2
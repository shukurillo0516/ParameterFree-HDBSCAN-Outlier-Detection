import csv
from scipy.stats import linregress
import pandas as pd

from utils import line_through_points
from intersect import intersection

from find_knee.knee import KneeThersholdFinder


# POLAR algorithm
def polar(glosh_scores, data):
    # glosh_scores_lab = sorted(
    #     [[score, idx, label] for idx, (score, label) in enumerate(zip(glosh_scores, data.iloc[:, -1]))],
    #     key=lambda x: x[0],
    # )
    sorted_glosh_scores = sorted(glosh_scores)

    # Find knee point
    indexes = list(range(len(sorted_glosh_scores)))
    kneeThres = KneeThersholdFinder(indexes, sorted_glosh_scores, clean_data=False)
    knee_idx, _ = kneeThres.find_knee()
    knee_idx = int(knee_idx[0])

    # Intersection and thresholds
    intersections = kneeThres.find_intersection_points()
    intersect = intersections[next(idx for idx, cntr in enumerate(indexes) if cntr == knee_idx)]

    x0, y0 = intersect[0][0], intersect[1][0]
    x2, y2 = (
        indexes[-1],
        linregress(indexes, sorted_glosh_scores)[0] * indexes[-1] + linregress(indexes, sorted_glosh_scores)[1],
    )

    x_line, y_line = line_through_points(x0, y0, x2, y2)
    _intersections = intersection(indexes, sorted_glosh_scores, x_line, y_line)
    x_intersects = _intersections[0]
    knee_threshold = sorted_glosh_scores[knee_idx]
    adjusted_knee_threshold = sorted_glosh_scores[round(x_intersects[-1])]

    return knee_threshold, adjusted_knee_threshold


def calc_polar_scores(data_info: list[dict]):
    """Data info must contain "Path" to datatest and "mpts_star" """
    results = []
    for i, row in enumerate(data_info):
        path = row["Path"]
        glosh_path = path.replace(".csv", "_outliers_python.csv")

        mpts_star = int(row["mpts_star"])

        try:
            glosh_scores = pd.read_csv(glosh_path)
        except Exception as err:
            print("$$$$$$$$$$$$$$$$$$")
            print()
            print("$$$$$$$$$$$$$$$$$$")
            continue
        mpts_star_scores = glosh_scores.iloc[:, mpts_star - 2]
        data = pd.read_csv(path)

        # try:
        true_outliers_count = data.iloc[:, -1].sum()
        threshold, addjusted_threshold = polar(mpts_star_scores, data)
        predicted_outliers_count = mpts_star_scores[mpts_star_scores >= threshold].count()
        top_n = predicted_outliers_count / true_outliers_count
        predicted_outliers_count_adj = mpts_star_scores[mpts_star_scores >= addjusted_threshold].count()
        top_n_adj = predicted_outliers_count_adj / true_outliers_count
        results.append(
            [
                path,
                mpts_star,
                true_outliers_count,
                predicted_outliers_count,
                predicted_outliers_count_adj,
                top_n,
                top_n_adj,
                threshold,
                addjusted_threshold,
            ]
        )
        # except Exception as err:
        #     continue

    return results


data_info = [
    # {"Path": "datasets/synthetic/local/banana_4.csv", "mpts_star": 11},
    {"Path": "datasets/synthetic/global/banana_1.csv", "mpts_star": 15},
    # {"Path": "datasets/synthetic/clumps/banana_1.csv", "mpts_star": 15},
]


def wrtite_top_n_results(results):
    header = [
        "Path",
        "mpts_star",
        "true_outliers_count",
        "predicted_outliers_count",
        "predicted_outliers_count_adj",
        "top_n",
        "top_n_adj",
        "threshold",
        "adj_threshold",
    ]
    with open("output/top_n_python.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)


if __name__ == "__main__":
    with open("output/best_mpts.csv", "r") as f:
        reader = csv.DictReader(f)
        polar_scores = calc_polar_scores(reader)
        wrtite_top_n_results(polar_scores)

import os
import csv
import numpy as np
import pandas as pd
import hdbscan
from auto_glosh import auto_glosh

SORTED = False


def calc_glosh_scores(file_paths: list[str]):
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        glosh_profiles = []
        for i in range(2, 102):
            clusterer = hdbscan.HDBSCAN(
                algorithm="generic",
                alpha=1.0,
                approx_min_span_tree=False,
                gen_min_span_tree=True,
                metric="euclidean",
                min_cluster_size=i,
                min_samples=i,
            )
            clusterer.fit(data.iloc[:, :-1])
            glosh_scores = np.nan_to_num(clusterer.outlier_scores_, nan=0)
            if SORTED:
                glosh_scores = np.sort(glosh_scores)
            glosh_profiles.append(glosh_scores)
        if SORTED:
            path = file_path.replace(".csv", "_outliers_python.csv")
        else:
            path = file_path.replace(".csv", "_not_sorted_outliers_python.csv")

        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(zip(*glosh_profiles))


# calc_glosh_scores(["datasets/synthetic/global/banana_1.csv"])


def main(paths: list[str]):
    for path in paths:
        profiles_path = path.replace(".csv", "_outliers_python.csv")
        data = pd.read_csv(path)
        profiles = pd.read_csv(profiles_path)

        mpts_star_glosh_scores = auto_glosh(profiles, data)
        # print(mpts_star_glosh_scores)


if __name__ == "__main__":
    with open("metadata.csv", "r") as f:
        reader = csv.DictReader(f)
        next(reader)
        paths = [os.path.join(row["Path"], row["Name"]) for row in reader]
        try:
            calc_glosh_scores(paths)
        except Exception as err:
            print(err)

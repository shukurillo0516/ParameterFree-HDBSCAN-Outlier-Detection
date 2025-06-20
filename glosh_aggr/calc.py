import os
import subprocess
import numpy as np
import pandas as pd
import hdbscan

from .utils import extract_java_outliers_data_from_txt

CURRENT_DIR = os.getcwd()


class GLOSHScore:
    def __init__(self, file_path: str, min_pts: int, min_clsize: int | None = None):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.min_pts = min_pts
        self.min_clsize = min_clsize if min_clsize else min_pts

    @staticmethod
    def get_tmp_path():
        return "glosh_aggr/tmp/glosh_scores.csv"

    def calc_py_outlier_scores(self) -> np.array:
        """"""
        clusterer = hdbscan.HDBSCAN(
            alpha=1.0,
            approx_min_span_tree=False,
            gen_min_span_tree=True,
            metric="euclidean",
            min_cluster_size=self.min_pts,
            min_samples=self.min_clsize,
            allow_single_cluster=False,
            match_reference_implementation=True,
        )
        clusterer.fit(self.data)
        return clusterer.outlier_scores_

    def calc_rust_outlier_scores(self, out_path: str | None = None) -> np.array:
        out_path = self.get_tmp_path() if out_path is None else out_path
        # Run Rust binary with args
        subprocess.run(
            [
                f"/home/shukurillo/lab/OD/ParameterFree-HDBSCAN-Outlier-Detection/glosh_aggr/rust_hdbscan",
                "--",
                f"--file_path={self.file_path}",
                f"--out_path={out_path}",
                f"--min_pts={self.min_pts}",
                f"--min_clsize={self.min_clsize}",
            ],
            check=True,
        )

        # Load the outlier scores from CSV
        outlier_scores = np.loadtxt(out_path, delimiter=",")

        return outlier_scores

    def calc_java_outlier_scores(self, out_path: str = "glosh_aggr/out_java/") -> np.array:
        # Run Java binary with args
        subprocess.run(
            [
                "java",
                "-cp",
                f"/home/shukurillo/lab/OD/ParameterFree-HDBSCAN-Outlier-Detection/glosh_aggr/elki-bundle-0.8.0.jar",
                "elki.application.KDDCLIApplication",
                "-dbc.in",
                self.file_path,
                "-dbc.parser",
                "NumberVectorLabelParser",
                "-algorithm",
                "outlier.clustering.GLOSH",
                "-hdbscan.minPts",
                str(self.min_pts),
                "-hdbscan.minclsize",
                str(self.min_clsize),
                "-out",
                out_path,
            ],
            check=True,
        )

        original_df = self.data.copy()
        # Load the outlier scores from CSV
        score_df = extract_java_outliers_data_from_txt(
            os.path.join(out_path, "GLOSH score Order.txt")
        )  # this is sorted by outlier score val
        merged_df = pd.merge(
            original_df, score_df, on=["x", "y"], how="left"  # ensures order of original_df is preserved
        )

        return merged_df["outlier_score"].to_numpy()


if __name__ == "__main__":
    glosh_score_calculator = GLOSHScore(
        file_path="/home/shukurillo/lab/OD/autoglosh-revisited/datasets/toy/toy.csv",
        min_pts=3,
    )
    print(glosh_score_calculator.calc_py_outlier_scores())
    print(glosh_score_calculator.calc_rust_outlier_scores())
    print(glosh_score_calculator.calc_java_outlier_scores())

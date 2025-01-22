# Unsupervised Parameter-free Outlier Detection using HDBSCAN* Outlier Profiles

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

***Published in 2024 IEEE International Conference on Big Data (BigData)***

**Paper URL:** [https://openreview.net/forum?id=lh6vOAHuvo](https://ieeexplore.ieee.org/abstract/document/10825917)

**Abstract:** In machine learning and data mining, outliers are data points that significantly differ from the dataset and often introduce irrelevant information that can induce bias in its statistics and models. Therefore, unsupervised methods are crucial to detect outliers if there is limited or no information about them. Global-Local Outlier Scores based on Hierarchies (GLOSH) is an unsupervised outlier detection method within HDBSCAN*, a state-of-the-art hierarchical clustering method. GLOSH estimates outlier scores for each data point by comparing its density to the highest density of the region they reside in the HDBSCAN* hierarchy. GLOSH may be sensitive to HDBSCAN*’s minpts parameter that influences density estimation. With limited knowledge about the data, choosing an appropriate minpts value beforehand is challenging as one or some minpts values may better represent the underlying cluster structure than others. Additionally, in the process of searching for "potential outliers", one has to define the number of outliers n a dataset has, which may be impractical and is often unknown. In this paper, we propose an unsupervised strategy to find the "best" minpts value, leveraging the range of GLOSH scores across minpts values to identify the value for which GLOSH scores can best identify outliers from the rest of the dataset. Moreover, we propose an unsupervised strategy to estimate a threshold for classifying points into inliers and (potential) outliers without the need to pre-define any value. Our experiments show that our strategies can automatically find the minpts value and threshold that yield the best or near best outlier detection results using GLOSH.

## Installation
Run the following command to install the dependencies:
```
# Python 3 is necessary use the tools in this repository
$ pip install -r requirements.txt
```

## Authors: 
1. Kushankur Ghosh
2. Murilo Coelho Naldi
3. J{\"o}rg Sander
4. Euijin Choo

## Citation: 
```
@inproceedings{ghosh2024unsupervised,
  title={Unsupervised Parameter-free Outlier Detection using HDBSCAN* Outlier Profiles},
  author={Ghosh, Kushankur and Naldi, Murilo Coelho and Sander, J{\"o}rg and Choo, Euijin},
  booktitle={2024 IEEE International Conference on Big Data (BigData)},
  pages={7021--7030},
  year={2024},
  organization={IEEE}
}
```

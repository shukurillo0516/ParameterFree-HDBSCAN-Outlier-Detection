from typing import Union
import numpy as np


class ElbowMptsFinder:
    """
    elbow minpts finder
    This tool searches for the point that has the maximum distance to a line
    passing through the first and last points of the given data set
    """

    def __init__(self, x_values: Union[list, np.ndarray], y_values: Union[list, np.ndarray, None] = None, clean_data: bool = True):
        """
        Initializing the ElbowMptsFinder with x and y data
        :param x_values: List or array of x-coordinates
        :param y_values: List or array of y-coordinates (optional if x_values is a 2D array)
        :param clean_data: Whether to remove duplicate values at the start and end of the data
        """
        if isinstance(x_values, np.ndarray) and y_values is None:
            assert x_values.shape[0] == 2, "x_values must be a 2D array with two rows"
            assert x_values.shape[1] > 2, "Too few points to find an elbow"
            self.data = x_values
        else:
            assert len(x_values) == len(y_values), "x_values and y_values must have the same length"
            assert len(x_values) > 2, "Too few points to find an elbow"
            self.data = np.vstack([x_values, y_values])

        if clean_data:
            self._clean_data()

        self.start_point = self.data.T[0]
        self.end_point = self.data.T[-1]
        self.line_vector = self.end_point - self.start_point
        self.start_point = np.expand_dims(self.start_point, axis=1)
        self.end_point = np.expand_dims(self.end_point, axis=1)
        self.line_vector = np.expand_dims(self.line_vector, axis=1)
        self.elbow = None

    def _clean_data(self):
        """
        Removing duplicate values at the beginning and end of the data.
        """
        start_index = 0
        end_index = len(self.data[1])
        first_value = self.data[1][0]
        last_value = self.data[1][-1]

        for i in range(end_index):
            if self.data[1][i] == first_value:
                start_index = i
            if self.data[1][end_index - i - 1] == last_value:
                end_index -= 1

        self.data = self.data[:, start_index:end_index]

    def find_elbow(self) -> np.ndarray:
        """
        Finding the elbow point in the data set
        :return: The coordinates of the elbow point
        """
        if self.elbow is not None:
            return self.elbow

        differences = self.start_point - self.data
        cross_products = np.abs(np.cross(self.line_vector, differences, axis=0))
        magnitudes = [np.linalg.norm(cross) for cross in cross_products]
        elbow_position = np.argmax(magnitudes)
        self.elbow = self.data[:, elbow_position]
        return self.elbow, magnitudes

    def find_intersection_points(self) -> list:
        """
        Finding the intersection points between the line and the data points
        :return: List of intersection points
        """
        intersections = []
        for i in range(len(self.data[0])):
            t = np.dot(
                np.array([self.data[0][i], self.data[1][i]]) - self.start_point.flatten(),
                self.line_vector.flatten()
            ) / np.dot(self.line_vector.flatten(), self.line_vector.flatten())

            intersection_point = self.start_point + t * self.line_vector
            intersections.append(intersection_point)

        return intersections

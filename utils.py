import numpy as np

# Helper functions
def line_through_points(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    x_line = np.linspace(min(x1, x2) - 1, max(x1, x2) + 1, 100)
    y_line = slope * (x_line - x1) + y1
    return x_line, y_line

def find_max_index(values):
    max_value = float('-inf')
    max_index = None
    supreme_max = float('-inf')
    for i, value in enumerate(values):
        if value > max_value:
            max_value = value
            max_index = i
            if value > supreme_max:
                supreme_max = value
        elif value > supreme_max / 2:
            max_value = value
            max_index = i
    return max_index
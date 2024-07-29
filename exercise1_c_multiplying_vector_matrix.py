import numpy as np


def compute_multiplying_matrix_vector(matrix, vector):
    result = np.dot(matrix, vector)

    return result


m = np.array([[-1, 1, 1], [0, -4, 9]])
v = np.array([0, 2, 1])
result = compute_multiplying_matrix_vector(m, v)
print(result)

import numpy as np


def compute_multiplying_matrix_vector(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)

    return result


m = np.array([[-1, 1, 1], [0, -4, 9]])
v = np.array([[0, 2, 1], [1, 2, 3], [1, 1, 1]])
result = compute_multiplying_matrix_vector(m, v)
print(result)

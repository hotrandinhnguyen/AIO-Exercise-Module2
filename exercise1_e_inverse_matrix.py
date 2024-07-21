import numpy as np


def matrix_inverse(matrix):
    result = np.linalg.inv(matrix)

    return result


m1 = np.array([[1, 2], [2, 3]])
result = matrix_inverse(m1)
print(result)
# check
i = m1 @ result
i = np.round(i, 2)
print("check")
print(i)

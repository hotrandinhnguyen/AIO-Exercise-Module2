import numpy as np


def compute_eigenvector_eigenvalues(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    return eigenvalues, eigenvectors


matrix = np.array([[1, 2], [2, 3]])
eigenvalues, eigenvectors = compute_eigenvector_eigenvalues(matrix)
print(eigenvalues)
print(eigenvectors)

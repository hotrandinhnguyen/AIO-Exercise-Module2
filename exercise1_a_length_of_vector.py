import numpy as np


def compute_vector_length(vector):
    norm = np.sqrt(np.sum([v**2 for v in vector]))
    # norm = np.linalog.norm(vector)
    return norm


vector = np.array([[1, 2, 3, 4]])
result = compute_vector_length([vector])
print(round(result, 2))

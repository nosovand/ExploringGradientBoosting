import numpy as np

X = np.array([
    [ 1.5, -2.3,  3.1],
    [-0.09269547780327612,  0.7, -1.2],
    [ 2.2, -1.9, -3.3],
    [-0.0018820165277906047,  1.8,  0.5]
])

sorted_indices = np.argsort(X[1:, :], axis=0)
sorted_values = np.take_along_axis(X[1:, :], sorted_indices, axis=0)

print("Original Array (excluding first row):")
print(X[1:, :])
print("Sorted Indices:")
print(sorted_indices)
print("Sorted Values:")
print(sorted_values)
from scipy.spatial import cKDTree
import numpy as np

# Example: 2D points
points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Build the k-d tree
tree = cKDTree(points)

# Query point
query_point = np.array([2, 3])

# Find the distance and index of the nearest neighbor
distance, index = tree.query(query_point)

print(f"Nearest neighbor index: {index}, Distance: {distance}")

# Find all pairs of points within a radius of 3
pairs = tree.query_pairs(3.0)
print(f"Pairs within radius: {list(pairs)}")

# Find the 2 nearest neighbors
distances, indices = tree.query(query_point, k=2)
print(f"Indices of 2 nearest neighbors: {indices}")
print(f"Distances: {distances}")

from scipy.spatial import cKDTree
import numpy as np

# Generate random 2D points
np.random.seed(42)
points = np.random.rand(100, 2)

# Build the tree
tree = cKDTree(points)

# Query point
query_point = np.array([0.5, 0.5])

# Find the nearest neighbor
distance, index = tree.query(query_point)

print(f"Nearest neighbor: {points[index]}, Distance: {distance}")

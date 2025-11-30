# visualizing embeddings

import numpy as np

print("EMBEDDING VISUALIZATION")
print()
print("embeddings are high-dimensional (100-300 dims)")
print("to visualize, reduce to 2D or 3D")
print()
print("methods:")
print("  - PCA: fast, linear")
print("  - t-SNE: non-linear, better clusters")
print("  - UMAP: newer, faster than t-SNE")
print()

# simple PCA for demonstration
def simple_pca_2d(embeddings):
    """reduce to 2D using PCA"""
    # center the data
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    # compute covariance
    cov = np.cov(centered.T)

    # eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # sort by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    top_2_vecs = eigenvectors[:, idx[:2]]

    # project
    return centered @ top_2_vecs

# example embeddings
np.random.seed(42)
words = ["cat", "dog", "kitten", "puppy", "car", "truck", "bus"]
embeddings = np.random.randn(len(words), 8)

# make similar words have similar embeddings
embeddings[0] = embeddings[1] + np.random.randn(8) * 0.1  # cat ~ dog
embeddings[2] = embeddings[0] + np.random.randn(8) * 0.1  # kitten ~ cat
embeddings[5] = embeddings[4] + np.random.randn(8) * 0.1  # truck ~ car
embeddings[6] = embeddings[4] + np.random.randn(8) * 0.1  # bus ~ car

projected = simple_pca_2d(embeddings)
print("2D projection (PCA):")
for word, (x, y) in zip(words, projected):
    print(f"  {word}: ({x.real:.2f}, {y.real:.2f})")

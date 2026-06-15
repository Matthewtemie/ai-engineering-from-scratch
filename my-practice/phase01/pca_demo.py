import numpy as np
import matplotlib.pyplot as plt

# Generate 200 fake 2D data points that mostly vary along a diagonal direction
np.random.seed(42)
n = 200
true_direction = np.array([3, 1])    # data clusters along this axis
noise_scale = 0.5
data = np.outer(np.random.randn(n), true_direction) + noise_scale * np.random.randn(n, 2)

# Center the data (subtract the mean)
data_centered = data - data.mean(axis=0)

# Covariance matrix — captures how the data varies
cov = np.cov(data_centered.T)
print("Covariance matrix:")
print(cov)

# Eigenvectors of the covariance matrix = principal components
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort so the biggest eigenvalue comes first
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print(f"\nEigenvalues (sorted): {eigenvalues}")
print(f"The first eigenvector points in the direction of most variation:")
print(f"  {eigenvectors[:, 0]}")
print(f"\nCompare to the true direction we built the data along: {true_direction / np.linalg.norm(true_direction)}")

# Plot the data and the principal directions
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.4, label='data')

# Draw eigenvectors scaled by sqrt of eigenvalues (so length reflects spread)
for i in range(2):
    vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 3
    ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, color='red', linewidth=2)
    ax.text(vec[0]*1.1, vec[1]*1.1, f'PC{i+1}', fontsize=14, color='red')

ax.set_aspect('equal')
ax.grid(True)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.legend()
ax.set_title('PCA: eigenvectors of the covariance matrix point along the data')
plt.savefig('pca.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nSaved pca.png")

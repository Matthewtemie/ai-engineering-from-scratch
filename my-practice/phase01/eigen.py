import numpy as np

# A simple stretch matrix — we already know the answer for this one
A = np.array([
    [2, 0],
    [0, 1],
])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (each column is one eigenvector):\n{eigenvectors}")

# Verify the eigenvalue equation A @ v = lambda * v for each
print("\nVerification — A @ v should equal lambda * v:")
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    left  = A @ v
    right = lam * v
    print(f"  Eigenvector {i+1}: A @ v = {left}, lambda * v = {right}")

# Now a less obvious matrix — find its hidden structure
print("\n" + "="*50)
print("A less obvious matrix:")
B = np.array([
    [4, 1],
    [2, 3],
])
print(B)
eigvals_B, eigvecs_B = np.linalg.eig(B)
print(f"\nEigenvalues: {eigvals_B}")
print(f"Eigenvectors (columns):\n{eigvecs_B}")
print("\nThese vectors are the directions B stretches without rotating.")

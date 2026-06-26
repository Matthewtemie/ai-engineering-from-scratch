import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate a small dataset from a wavy curve plus noise
n = 30
X = np.sort(np.random.uniform(0, 1, n))
y_true = np.sin(2 * np.pi * X)              # a wave
y = y_true + np.random.normal(0, 0.2, n)    # add noise

# Split 70/30
split = int(0.7 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Try fitting polynomials of different degrees
# A polynomial of degree d has the form y = a0 + a1*x + a2*x^2 + ... + ad*x^d
# Low degree = simple model; high degree = complex model
degrees = [1, 3, 10, 25]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, degree in zip(axes, degrees):
    # numpy fits a polynomial of the requested degree to the training data
    coeffs = np.polyfit(X_train, y_train, degree)
    
    # Compute predictions on training and test sets
    y_train_pred = np.polyval(coeffs, X_train)
    y_test_pred  = np.polyval(coeffs, X_test)
    
    # Mean squared error (a common regression metric)
    train_mse = ((y_train_pred - y_train) ** 2).mean()
    test_mse  = ((y_test_pred  - y_test)  ** 2).mean()
    
    # Plot the data and the fitted curve
    X_dense = np.linspace(0, 1, 200)
    ax.scatter(X_train, y_train, color='blue', label='train', s=30)
    ax.scatter(X_test, y_test, color='red', marker='x', label='test', s=50)
    ax.plot(X_dense, np.polyval(coeffs, X_dense), 'g-', alpha=0.7, label=f'degree {degree}')
    ax.set_title(f"Degree {degree}\ntrain MSE={train_mse:.3f}  test MSE={test_mse:.3f}")
    ax.set_ylim(-2, 2)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting.png', dpi=100, bbox_inches='tight')
plt.show()

print("Compare the train and test MSE for each degree.")
print("Notice: increasing complexity eventually HELPS train error but HURTS test error.")
print("That's overfitting in one picture.")
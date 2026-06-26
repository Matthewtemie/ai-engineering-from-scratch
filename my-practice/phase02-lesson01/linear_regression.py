import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Step 1: generate synthetic data with a known truth ----------
# We "know" the true relationship: y = 3*x1 + 1.5*x2 - 2*x3 + 5 + noise
# That way we can check if our learned weights recover the truth.
n_samples = 200
n_features = 3
true_w = np.array([3.0, 1.5, -2.0])
true_b = 5.0

X = np.random.randn(n_samples, n_features)              # 200 examples, 3 features each
noise = np.random.normal(0, 0.5, size=n_samples)        # small Gaussian noise
y = X @ true_w + true_b + noise                          # the true labels with noise

# ---------- Step 2: train/test split (Lesson 15 habit) ----------
indices = np.random.permutation(n_samples)
split = int(0.8 * n_samples)
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

print(f"Training set: {X_train.shape[0]} examples")
print(f"Test set:     {X_test.shape[0]} examples")

# ---------- Step 3: implement linear regression with gradient descent ----------
def train(X, y, learning_rate=0.05, epochs=500):
    n, d = X.shape
    w = np.zeros(d)              # start with zeros — could also start random
    b = 0.0
    losses = []

    for epoch in range(epochs):
        # Forward pass: predict
        y_pred = X @ w + b
        
        # Loss: mean squared error
        error = y_pred - y
        loss = (error ** 2).mean()
        losses.append(loss)
        
        # Backward pass: compute gradients of MSE with respect to w and b
        # dL/dw = (2/n) * X^T @ error
        # dL/db = (2/n) * sum(error)
        grad_w = (2 / n) * X.T @ error
        grad_b = (2 / n) * error.sum()
        
        # Update
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
    
    return w, b, losses

w_learned, b_learned, losses = train(X_train, y_train)

print(f"\nTrue weights:    {true_w}")
print(f"Learned weights: {w_learned.round(3)}")
print(f"True bias:    {true_b}")
print(f"Learned bias: {b_learned:.3f}")

# ---------- Step 4: evaluate on the test set ----------
def mse(X, y, w, b):
    y_pred = X @ w + b
    return ((y_pred - y) ** 2).mean()

train_mse = mse(X_train, y_train, w_learned, b_learned)
test_mse  = mse(X_test,  y_test,  w_learned, b_learned)
print(f"\nTrain MSE: {train_mse:.4f}")
print(f"Test  MSE: {test_mse:.4f}")

# ---------- Step 5: plot the loss curve ----------
plt.figure(figsize=(8, 5))
plt.plot(losses, 'b-')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Loss curve — gradient descent finding the minimum')
plt.grid(True, alpha=0.3)
plt.savefig('loss_curve.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved loss_curve.png")

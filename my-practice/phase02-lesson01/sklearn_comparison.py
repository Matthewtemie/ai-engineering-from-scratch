import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Same synthetic data as before
np.random.seed(42)
n_samples = 200
true_w = np.array([3.0, 1.5, -2.0])
true_b = 5.0
X = np.random.randn(n_samples, 3)
y = X @ true_w + true_b + np.random.normal(0, 0.5, size=n_samples)

# sklearn's built-in train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train: just two lines
model = LinearRegression()
model.fit(X_train, y_train)

# Inspect what was learned
print(f"True weights:    {true_w}")
print(f"sklearn weights: {model.coef_.round(3)}")
print(f"True bias:    {true_b}")
print(f"sklearn bias: {model.intercept_:.3f}")

# Evaluate
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)
print(f"\nTrain MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Test  MSE: {mean_squared_error(y_test,  y_test_pred):.4f}")
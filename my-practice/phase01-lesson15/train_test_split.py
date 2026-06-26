import numpy as np

np.random.seed(42)

# Make some fake data: 100 examples, each with one feature
# True relationship: y = 2x + 3 + noise
n_total = 100
X = np.random.uniform(0, 10, size=n_total)
y = 2 * X + 3 + np.random.normal(0, 1, size=n_total)

# Shuffle indices so train/test isn't biased by data ordering
indices = np.arange(n_total)
np.random.shuffle(indices)

# 80/20 split
split = int(0.8 * n_total)
train_idx = indices[:split]
test_idx  = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"Total examples:   {n_total}")
print(f"Training set:     {len(X_train)} examples")
print(f"Test set:         {len(X_test)} examples")
print(f"\nTraining set X range: {X_train.min():.2f} to {X_train.max():.2f}")
print(f"Test set X range:     {X_test.min():.2f} to {X_test.max():.2f}")
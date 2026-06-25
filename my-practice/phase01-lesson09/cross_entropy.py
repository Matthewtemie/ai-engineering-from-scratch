import numpy as np

def cross_entropy(true_dist, predicted_dist):
    """Cross-entropy: -sum(p_true * log(p_predicted))."""
    p = np.array(true_dist, dtype=float)
    q = np.array(predicted_dist, dtype=float)
    return -np.sum(p * np.log(q + 1e-12))

# True label: it's a cat. One-hot vector across [cat, dog, bird]
true_label = [1.0, 0.0, 0.0]

predictions = [
    ("Confident, correct",    [0.99, 0.005, 0.005]),
    ("Pretty confident",      [0.70, 0.20, 0.10]),
    ("Uncertain (uniform)",   [0.34, 0.33, 0.33]),
    ("Confidently wrong",     [0.01, 0.98, 0.01]),
    ("Catastrophically wrong",[0.001, 0.999, 0.0]),
]

print(f"{'Prediction':<28} | {'Cross-entropy':>13}")
print("-" * 45)
for name, pred in predictions:
    ce = cross_entropy(true_label, pred)
    print(f"{name:<28} | {ce:>13.4f}")

print("\nKey observation: a CONFIDENT wrong answer is much worse than an UNCERTAIN one.")
print("This is the principle that trains every classifier you've ever used.")
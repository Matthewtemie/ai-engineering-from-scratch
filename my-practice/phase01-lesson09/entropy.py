import numpy as np
import matplotlib.pyplot as plt

def entropy(probs):
    """Shannon entropy: -sum(p * log(p)). Uses natural log."""
    probs = np.array(probs, dtype=float)
    # The 1e-12 avoids log(0) which would be -infinity
    return -np.sum(probs * np.log(probs + 1e-12))

# Compare distributions
distributions = [
    ("Fair coin",              [0.5, 0.5]),
    ("Weighted coin 99/1",     [0.99, 0.01]),
    ("Almost-certain coin",    [0.999, 0.001]),
    ("Fair 6-sided die",       [1/6]*6),
    ("Loaded die",             [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("Three-class uniform",    [1/3, 1/3, 1/3]),
    ("Three-class confident",  [0.98, 0.01, 0.01]),
]

print(f"{'Distribution':<28} | {'Entropy (nats)':>14}")
print("-" * 46)
for name, dist in distributions:
    h = entropy(dist)
    print(f"{name:<28} | {h:>14.4f}")

# Visualize: entropy of a coin as p varies from 0 to 1
ps = np.linspace(0.001, 0.999, 200)
entropies = [entropy([p, 1-p]) for p in ps]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ps, entropies, 'b-', linewidth=2)
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='maximum at p=0.5')
ax.set_xlabel('P(heads)')
ax.set_ylabel('Entropy (nats)')
ax.set_title('Entropy of a coin as it goes from certain → uncertain → certain')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('entropy_coin.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nSaved entropy_coin.png")
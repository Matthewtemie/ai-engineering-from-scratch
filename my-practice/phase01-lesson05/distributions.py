import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)   # makes random numbers reproducible

# Draw 10,000 samples from three different distributions
n = 10_000
uniform_samples = np.random.uniform(low=0, high=1, size=n)
normal_samples  = np.random.normal(loc=0, scale=1, size=n)    # mean=0, std=1
bernoulli_samples = np.random.binomial(n=1, p=0.3, size=n)    # p=0.3 of "1"

# Summary statistics
def summarize(name, samples):
    print(f"{name}:")
    print(f"  mean     = {samples.mean():.4f}")
    print(f"  variance = {samples.var():.4f}")
    print(f"  std dev  = {samples.std():.4f}")
    print(f"  min, max = {samples.min():.4f}, {samples.max():.4f}")
    print()

summarize("Uniform [0, 1]", uniform_samples)
summarize("Normal (mean=0, std=1)", normal_samples)
summarize("Bernoulli (p=0.3)", bernoulli_samples)

# Plot all three side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(uniform_samples, bins=40, color='steelblue', edgecolor='white')
axes[0].set_title("Uniform — every value equally likely")
axes[0].set_xlabel("value")

axes[1].hist(normal_samples, bins=40, color='seagreen', edgecolor='white')
axes[1].set_title("Normal — bell curve around the mean")
axes[1].set_xlabel("value")

axes[2].hist(bernoulli_samples, bins=[-0.25, 0.25, 0.75, 1.25],
             color='indianred', edgecolor='white')
axes[2].set_title("Bernoulli — yes/no with p=0.3")
axes[2].set_xlabel("value")
axes[2].set_xticks([0, 1])

plt.tight_layout()
plt.savefig('distributions.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved distributions.png")
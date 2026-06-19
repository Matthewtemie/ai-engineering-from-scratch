import numpy as np
import matplotlib.pyplot as plt

# True die — fair 6-sided. Expected value (mean) = 3.5
np.random.seed(42)
n_rolls = 5000
rolls = np.random.randint(1, 7, size=n_rolls)   # integers 1-6

# Running average after each roll
running_avg = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

# Print a few checkpoints
for n in [10, 100, 1000, 5000]:
    print(f"After {n:>5} rolls, average = {running_avg[n-1]:.4f}  (true mean = 3.5)")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(running_avg, color='steelblue', linewidth=1.5, label='running average')
ax.axhline(3.5, color='red', linestyle='--', label='true mean (3.5)')
ax.set_xlabel('number of rolls')
ax.set_ylabel('running average')
ax.set_title('Law of Large Numbers — averages converge to the true mean')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('law_of_large_numbers.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved law_of_large_numbers.png")
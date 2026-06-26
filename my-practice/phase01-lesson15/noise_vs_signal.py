import numpy as np

np.random.seed(0)

# Imagine we're comparing two model designs by running each one many times
# with different random seeds and seeing the spread of scores.

# Model A: average accuracy 0.81 with some run-to-run variation
# Model B: average accuracy 0.83 with the same variation
# Are they actually different, or is the 0.02 gap inside the noise?

n_runs = 30
model_a_scores = np.random.normal(loc=0.81, scale=0.03, size=n_runs)
model_b_scores = np.random.normal(loc=0.83, scale=0.03, size=n_runs)

print(f"Model A: mean={model_a_scores.mean():.4f}, std={model_a_scores.std():.4f}")
print(f"Model B: mean={model_b_scores.mean():.4f}, std={model_b_scores.std():.4f}")
print(f"Mean difference: {model_b_scores.mean() - model_a_scores.mean():+.4f}")

# Standard error of the difference
se_a = model_a_scores.std() / np.sqrt(n_runs)
se_b = model_b_scores.std() / np.sqrt(n_runs)
se_diff = np.sqrt(se_a**2 + se_b**2)

# Rough 95% confidence interval around the difference
diff = model_b_scores.mean() - model_a_scores.mean()
ci_lower = diff - 1.96 * se_diff
ci_upper = diff + 1.96 * se_diff

print(f"\n95% confidence interval for the difference: [{ci_lower:+.4f}, {ci_upper:+.4f}]")

if ci_lower > 0:
    print("Conclusion: B is RELIABLY better than A.")
elif ci_upper < 0:
    print("Conclusion: A is RELIABLY better than B.")
else:
    print("Conclusion: The difference is NOT reliably distinguishable from noise.")

# Try cranking up the noise to see when the test would say "uncertain"
print("\n" + "="*60)
print("Now what if there's more run-to-run variation (std=0.06)?")
np.random.seed(0)
model_a_noisy = np.random.normal(loc=0.81, scale=0.06, size=n_runs)
model_b_noisy = np.random.normal(loc=0.83, scale=0.06, size=n_runs)
diff_noisy = model_b_noisy.mean() - model_a_noisy.mean()
se_diff_noisy = np.sqrt(2) * 0.06 / np.sqrt(n_runs)
ci_lo = diff_noisy - 1.96 * se_diff_noisy
ci_hi = diff_noisy + 1.96 * se_diff_noisy
print(f"Mean difference: {diff_noisy:+.4f}")
print(f"95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
if ci_lo > 0 or ci_hi < 0:
    print("Difference is real.")
else:
    print("Difference is INSIDE the noise — we cannot conclude B beats A.")
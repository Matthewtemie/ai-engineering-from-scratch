def bayes(prior, likelihood_given_true, likelihood_given_false):
    """
    P(A | B) = P(B|A) * P(A) / P(B)
    where P(B) = P(B|A)*P(A) + P(B|not A)*P(not A)
    """
    p_not_a = 1 - prior
    marginal = likelihood_given_true * prior + likelihood_given_false * p_not_a
    posterior = (likelihood_given_true * prior) / marginal
    return posterior

# The disease test from above
prior = 0.001                  # base rate of disease
likelihood_given_true = 0.99  # P(test+ | disease)
likelihood_given_false = 0.01  # P(test+ | no disease)

posterior = bayes(prior, likelihood_given_true, likelihood_given_false)
print(f"Disease test scenario:")
print(f"  Prior (base rate):           {prior:.3%}")
print(f"  Test true positive rate:     {likelihood_given_true:.1%}")
print(f"  Test false positive rate:    {likelihood_given_false:.1%}")
print(f"  Posterior P(disease | +):    {posterior:.3%}")

print("\n" + "="*55)
print("What if the disease is more common?")
for prior in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]:
    post = bayes(prior, 0.99, 0.01)
    print(f"  Prior = {prior:>6.1%}  →  Posterior after + test = {post:.1%}")
def update_belief(prior, p_evidence_given_true, p_evidence_given_false):
    """One Bayesian update step."""
    p_not = 1 - prior
    marginal = p_evidence_given_true * prior + p_evidence_given_false * p_not
    return (p_evidence_given_true * prior) / marginal


# Start with a baseline: 30% of emails are spam (our prior)
belief_spam = 0.30
print(f"Starting belief (prior): {belief_spam:.1%}\n")

# Each piece of evidence: how often this word appears in spam vs. non-spam
evidence = [
    ("contains the word 'lottery'",     0.40, 0.01),
    ("contains a suspicious URL",        0.60, 0.05),
    ("contains 'click here'",           0.50, 0.10),
    ("written in all caps",              0.30, 0.02),
    ("contains the word 'meeting'",     0.05, 0.40),
]

for description, p_in_spam, p_in_ham in evidence:
    new_belief = update_belief(belief_spam, p_in_spam, p_in_ham)
    print(f"After observing: {description}")
    print(f"  Belief jumps: {belief_spam:.1%}  →  {new_belief:.1%}")
    belief_spam = new_belief
    print()

print(f"Final belief this email is spam: {belief_spam:.2%}")
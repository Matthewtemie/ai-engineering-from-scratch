import numpy as np

def softmax(scores):
    """Turn arbitrary scores into a probability distribution."""
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()

# Imagine the model thinks these are the "scores" for what the next word is
words = ["the", "a", "an", "banana"]
scores = np.array([5.2, 4.5, 3.1, -2.0])

probabilities = softmax(scores)

print(f"{'word':>10} | {'score':>6} | {'probability':>12}")
print("-" * 36)
for w, s, p in zip(words, scores, probabilities):
    print(f"{w:>10} | {s:>6.2f} | {p:>12.4f}")

print(f"\nProbabilities sum to: {probabilities.sum():.4f}  (should be 1.0)")
print("\nThis is literally how every LLM picks the next word.")
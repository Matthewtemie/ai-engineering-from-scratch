import torch
import torch.nn.functional as F

# Imagine these are the raw scores ("logits") from a model's final layer
# for a single example, across 3 classes [cat, dog, bird]
logits = torch.tensor([2.5, 0.5, -1.0], requires_grad=True)

# The true label: "cat" is class index 0
true_label = torch.tensor([0])

# PyTorch's cross_entropy combines softmax + cross-entropy in one stable call
loss = F.cross_entropy(logits.unsqueeze(0), true_label)
print(f"Logits: {logits.tolist()}")
print(f"Loss: {loss.item():.4f}")

# Now compute the gradient — this is what training uses
loss.backward()
print(f"\nGradient with respect to logits: {logits.grad.tolist()}")
print("\nThese gradients tell the optimizer: 'nudge the cat-logit UP,")
print("and nudge the dog and bird logits DOWN, to reduce the loss.'")
print("That's a single training step on a single example.")
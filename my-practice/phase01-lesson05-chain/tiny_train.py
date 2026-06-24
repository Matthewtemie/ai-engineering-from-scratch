import torch

# A tiny "model": y = w * x + b, with parameters w and b to be learned
# We'll train it to learn the relationship y = 2x + 3 from data

# Initialize parameters randomly
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Training data: x values and the true outputs (y = 2x + 3)
x_data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0, 11.0])

learning_rate = 0.05

print(f"{'step':>4} | {'w':>7} | {'b':>7} | {'loss':>8}")
print("-" * 36)

for step in range(50):
    # Forward pass: compute predictions
    y_pred = w * x_data + b
    # Loss: average squared error
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward pass: PyTorch computes dloss/dw and dloss/db
    loss.backward()
    
    # Gradient descent update — wrap in no_grad so we don't track this op
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        # Clear gradients before the next step (PyTorch accumulates them)
        w.grad.zero_()
        b.grad.zero_()
    
    if step % 5 == 0:
        print(f"{step:>4} | {w.item():>7.4f} | {b.item():>7.4f} | {loss.item():>8.4f}")

print(f"\nLearned: y = {w.item():.3f} * x + {b.item():.3f}")
print("True:    y = 2.000 * x + 3.000")
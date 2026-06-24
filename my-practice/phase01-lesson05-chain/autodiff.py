import torch

# Same composed function as before: y = (3x + 1)^2
# But now we let PyTorch track operations on x

# requires_grad=True tells PyTorch: "watch this tensor, I'll want gradients later"
x = torch.tensor(2.0, requires_grad=True)

# Build the computation — PyTorch silently records every step
g = 3 * x + 1
y = g ** 2

print(f"x = {x.item()}")
print(f"g(x) = {g.item()}")
print(f"y = f(g(x)) = {y.item()}")

# Ask PyTorch for the gradient. It walks backward through the recorded ops.
y.backward()

print(f"\nPyTorch computed dy/dx = {x.grad.item()}")
print(f"By chain rule, dy/dx at x=2 should be 6(3x+1) = 6*7 = 42")
print(f"Match? {abs(x.grad.item() - 42) < 0.0001}")

print("\n" + "="*50)
print("Now try a function so complex no one would derive it by hand:")

x2 = torch.tensor(1.5, requires_grad=True)
y2 = torch.sin(x2) * torch.exp(x2 ** 2) + torch.log(x2 + 3)
y2.backward()
print(f"\nAt x = {x2.item()}, the derivative of that monster is: {x2.grad.item():.6f}")
print("You did not derive that by hand. PyTorch did it instantly.")
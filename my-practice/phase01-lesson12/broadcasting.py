import torch

# Example 1: bias vector added to a batch of activations
print("="*60)
print("Example 1: Adding a bias vector to a matrix")
print("="*60)
activations = torch.tensor([
    [1.0, 2, 3, 4],     # row 1: outputs of layer for first example
    [5.0, 6, 7, 8],     # row 2: second example
    [9.0, 10, 11, 12],  # row 3: third example
])
bias = torch.tensor([0.1, 0.2, 0.3, 0.4])   # one bias per neuron

print(f"activations shape: {tuple(activations.shape)}")
print(f"bias shape:        {tuple(bias.shape)}")
result = activations + bias    # bias broadcasts across rows
print(f"result shape:      {tuple(result.shape)}")
print(f"result:\n{result}")
print("\nThe SAME bias was added to EVERY row. That's broadcasting.")

# Example 2: subtracting per-row averages (centering data)
print("\n" + "="*60)
print("Example 2: Centering each row by subtracting its mean")
print("="*60)
data = torch.tensor([
    [10.0, 20, 30],
    [1.0, 2, 3],
    [100.0, 200, 300],
])
row_means = data.mean(dim=1, keepdim=True)  # keepdim=True preserves the column
print(f"data shape:      {tuple(data.shape)}")
print(f"row_means shape: {tuple(row_means.shape)}  ← note the (3,1) column")
print(f"row_means:\n{row_means}")
centered = data - row_means
print(f"centered (each row now has mean 0):\n{centered}")
print(f"new row means: {centered.mean(dim=1).tolist()}")

# Example 3: a broadcasting FAILURE (this is the common bug)
print("\n" + "="*60)
print("Example 3: Broadcasting failure — common shape bug")
print("="*60)
A = torch.zeros(3, 4)
B = torch.zeros(3)        # wrong shape if we want per-row broadcast
print(f"A shape: {tuple(A.shape)}")
print(f"B shape: {tuple(B.shape)}")
try:
    A + B   # this FAILS because (3,) tries to align with the 4-dim, not the 3-dim
except RuntimeError as e:
    print(f"ERROR: {e}")
print("FIX: use B.reshape(3, 1) so it broadcasts down the rows.")

C = torch.zeros(3, 1)
print(f"\nA + C with C.shape={tuple(C.shape)} works: result shape {tuple((A + C).shape)}")
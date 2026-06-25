import torch

# Make tensors of various ranks
scalar = torch.tensor(5.0)
vector = torch.tensor([1.0, 2, 3, 4])
matrix = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
batch  = torch.randn(8, 3, 224, 224)   # 8 fake color images, 3 channels, 224x224

print("Tensor shapes and ranks:")
for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("batch", batch)]:
    print(f"  {name:7s}  shape={tuple(t.shape):<22}  ndim={t.ndim}  total elements={t.numel()}")

# Reshape demonstrations
print("\nReshape demos:")
x = torch.arange(12)               # [0, 1, 2, ..., 11]
print(f"  Original (shape {tuple(x.shape)}):\n  {x}")
print(f"  Reshape to (3, 4):\n  {x.reshape(3, 4)}")
print(f"  Reshape to (2, 2, 3):\n  {x.reshape(2, 2, 3)}")
print(f"  Reshape with -1 inference, (4, -1):\n  {x.reshape(4, -1)}")

# Transpose
print("\nTranspose:")
m = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"  Original (shape {tuple(m.shape)}):\n  {m}")
print(f"  Transposed (shape {tuple(m.T.shape)}):\n  {m.T}")
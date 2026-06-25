import torch

# Imagine a batch of 8 examples, each with 10 query vectors of dimension 64
# AND 10 key vectors of dimension 64
batch_size = 8
seq_len = 10
dim = 64

Q = torch.randn(batch_size, seq_len, dim)   # (8, 10, 64)
K = torch.randn(batch_size, seq_len, dim)   # (8, 10, 64)

# We want to compute, for each batch, Q @ K.T to get (8, 10, 10) attention scores
# K.transpose(-2, -1) swaps the LAST two dims, turning (8,10,64) into (8,64,10)
scores = Q @ K.transpose(-2, -1)

print(f"Q shape:               {tuple(Q.shape)}")
print(f"K shape:               {tuple(K.shape)}")
print(f"K.transpose(-2,-1):    {tuple(K.transpose(-2, -1).shape)}")
print(f"Q @ K.transpose(-2,-1) shape: {tuple(scores.shape)}")
print(f"\nFor each of the 8 examples, we got a 10x10 matrix of similarity scores.")
print(f"That's the inner mechanic of attention — and you just did it in ONE line.")
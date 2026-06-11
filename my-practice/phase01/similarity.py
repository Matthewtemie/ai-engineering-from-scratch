import numpy as np

# Pretend these are tiny "word embeddings" — each dimension represents
# some abstract concept like [animal-ness, vehicle-ness, fluffiness]
cat   = np.array([0.9, 0.1, 0.8])
dog   = np.array([0.85, 0.15, 0.75])
car   = np.array([0.1, 0.95, 0.2])
truck = np.array([0.05, 0.92, 0.1])

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    return dot / (np.linalg.norm(a) * np.linalg.norm(b))

pairs = [
    ("cat",  "dog",   cat,   dog),
    ("cat",  "car",   cat,   car),
    ("car",  "truck", car,   truck),
    ("dog",  "truck", dog,   truck),
]

print("Cosine similarity between pairs:")
print("-" * 40)
for name_a, name_b, a, b in pairs:
    sim = cosine_similarity(a, b)
    print(f"  {name_a:5s} vs {name_b:5s}: {sim:+.4f}")

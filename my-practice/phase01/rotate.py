import numpy as np
import matplotlib.pyplot as plt

# A square: four corner points as 2D vectors, stacked into a 2×4 matrix
# Each column is one corner.
square = np.array([
    [0, 1, 1, 0],   # x-coordinates of the 4 corners
    [0, 0, 1, 1],   # y-coordinates
])

# Rotation matrix: rotate by 45 degrees
angle = np.pi / 4   # 45 degrees in radians
R = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)],
])

# Apply the transformation: matrix × matrix
rotated = R @ square    # the @ symbol means matrix multiplication in Python

print("Original corners (each column = one corner):")
print(square)
print("\nRotated corners:")
print(rotated)

# Draw both shapes. Close the polygons by repeating the first column at the end.
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.append(square[0],  square[0,0]),  np.append(square[1],  square[1,0]),  'b-o', label='original')
ax.plot(np.append(rotated[0], rotated[0,0]), np.append(rotated[1], rotated[1,0]), 'r-o', label='rotated 45°')
ax.set_aspect('equal')
ax.grid(True)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.legend()
ax.set_title('A matrix rotates a square')
plt.savefig('rotation.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nSaved rotation.png")

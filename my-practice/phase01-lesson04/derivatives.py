import numpy as np
import matplotlib.pyplot as plt

# The function we'll explore
def f(x):
    return x**2

# Numerical derivative: slope using a tiny step h
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# Test at a few points
print("f(x) = x^2")
print(f"{'x':>4} | {'f(x)':>6} | {'derivative':>10} | {'expected (2x)':>13}")
print("-" * 45)
for x in [-2, -1, 0, 1, 2, 3]:
    d = numerical_derivative(f, x)
    print(f"{x:>4} | {f(x):>6} | {d:>10.5f} | {2*x:>13}")

# Plot the curve, and the slope at one specific point
x_vals = np.linspace(-3, 3, 100)
y_vals = f(x_vals)

x_point = 2
y_point = f(x_point)
slope = numerical_derivative(f, x_point)

# Tangent line: passes through (x_point, y_point) with the slope above
tangent_x = np.linspace(x_point - 1.5, x_point + 1.5, 50)
tangent_y = slope * (tangent_x - x_point) + y_point

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(x_vals, y_vals, 'b-', label='f(x) = x²')
ax.plot(tangent_x, tangent_y, 'r--', label=f'slope at x={x_point} is {slope:.2f}')
ax.plot(x_point, y_point, 'ro', markersize=10)
ax.grid(True)
ax.legend()
ax.set_title('The derivative is the slope of the tangent line')
plt.savefig('derivative.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nSaved derivative.png")

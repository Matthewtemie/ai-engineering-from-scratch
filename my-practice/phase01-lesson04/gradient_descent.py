import numpy as np
import matplotlib.pyplot as plt

# A simple bowl-shaped function with a clear minimum at x = 0
def loss(x):
    return x**2

def gradient(x, h=1e-5):
    return (loss(x + h) - loss(x)) / h

# Gradient descent loop
x = 4.0                # starting position (terrible — we want to find x=0)
learning_rate = 0.1    # step size
history = [x]

print(f"{'step':>4} | {'x':>9} | {'loss(x)':>9} | {'gradient':>9}")
print("-" * 40)
for step in range(30):
    g = gradient(x)
    print(f"{step:>4} | {x:>9.5f} | {loss(x):>9.5f} | {g:>9.5f}")
    x = x - learning_rate * g    # the gradient descent update rule
    history.append(x)

print(f"\nFinal x ≈ {x:.5f}  (true minimum is at x = 0)")

# Visualize the descent
x_vals = np.linspace(-5, 5, 200)
y_vals = loss(x_vals)
history = np.array(history)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_vals, y_vals, 'b-', alpha=0.4, label='loss(x) = x²')
ax.plot(history, loss(history), 'ro-', markersize=6, label='descent path')
ax.plot(history[0], loss(history[0]), 'gs', markersize=12, label='start')
ax.plot(history[-1], loss(history[-1]), 'k*', markersize=15, label='end')
ax.grid(True)
ax.legend()
ax.set_title('Gradient descent walking to the minimum')
ax.set_xlabel('x')
ax.set_ylabel('loss')
plt.savefig('gradient_descent.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved gradient_descent.png")

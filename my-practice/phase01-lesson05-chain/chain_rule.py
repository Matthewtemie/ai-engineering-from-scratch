import numpy as np

# Two functions, composed: y = f(g(x)) where g(x) = 3x+1, f(g) = g^2
def g(x):
    return 3 * x + 1

def f(g_val):
    return g_val ** 2

def composed(x):
    return f(g(x))

# Numerical derivative — black-box, doesn't know about composition
def numerical_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x)) / h

# Symbolic derivative via the chain rule:
# dy/dx = dy/dg * dg/dx = 2g * 3 = 6g = 6(3x+1)
def chain_rule_derivative(x):
    dg_dx = 3
    dy_dg = 2 * g(x)
    return dy_dg * dg_dx

print(f"{'x':>4} | {'numerical':>10} | {'chain rule':>10} | {'match?':>7}")
print("-" * 45)
for x in [0, 1, 2, 3, -2]:
    num = numerical_derivative(composed, x)
    chn = chain_rule_derivative(x)
    match = "✓" if abs(num - chn) < 0.01 else "✗"
    print(f"{x:>4} | {num:>10.4f} | {chn:>10.4f} | {match:>7}")
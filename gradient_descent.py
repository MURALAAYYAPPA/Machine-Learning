import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("test_scores.csv")

# Independent and dependent variables
x = df['math'].values
y = df['cs'].values

def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.0002

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr

        cost = (1/n) * np.sum((y - y_predicted) ** 2)

        md = -(2/n) * np.sum(x * (y - y_predicted))
        bd = -(2/n) * np.sum(y - y_predicted)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        if i % 100 == 0:
            print(f"Iteration {i}: m={m_curr:.4f}, b={b_curr:.4f}, cost={cost:.4f}")

    return m_curr, b_curr

# Run gradient descent
m, b = gradient_descent(x, y)

print("\nFinal values:")
print("Slope (m):", m)
print("Intercept (b):", b)

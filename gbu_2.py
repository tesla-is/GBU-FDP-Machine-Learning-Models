import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target


plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], c = "r", label = "Setosa")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], c = "g", label = "Versicolor")
plt.scatter(X[:, 0][y == 2], X[:, 1][y == 2], c = "b", label = "verginica")
plt.legend()
plt.show()

plt.scatter(X[:, 2][y == 0], X[:, 3][y == 0], c = "r", label = "Setosa")
plt.scatter(X[:, 2][y == 1], X[:, 3][y == 1], c = "g", label = "Versicolor")
plt.scatter(X[:, 2][y == 2], X[:, 3][y == 2], c = "b", label = "verginica")
plt.legend()
plt.show()



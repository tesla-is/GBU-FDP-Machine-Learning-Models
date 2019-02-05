import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

from sklearn.datasets import fetch_mldata
dataset = fetch_mldata('MNIST original')

X = dataset.data
some_digit = X[33220]
some_digit_image = some_digit.reshape(28, 28)
y = dataset.target

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 10)
dtf.fit(X_train, y_train)

dtf.score(X_train, y_train)
dtf.score(X_test, y_test)
dtf.score(X, y)

dtf.predict(X[[567, 69998, 33220], 0:784])










































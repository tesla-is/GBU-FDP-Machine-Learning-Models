import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset/DemographicData.csv')

x = np.arange(-10, 10, 0.01)
y = 1 / (1 + np.power(np.e, -x))

plt.plot(x, y)
plt.show()

x = dataset.iloc[:, 2].values
y = dataset.iloc[:, 3].values
z = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
z = labelencoder.fit_transform(z)

labelencoder.classes_



plt.scatter(x[z == 0], y[z == 0], c = "r", label = "High Income")
plt.scatter(x[z == 1], y[z == 1], c = "g", label = "Low Income")
plt.scatter(x[z == 2], y[z == 2], c = "b", label = "Lower Middle Income")
plt.scatter(x[z == 3], y[z == 3], c = "cyan", label = "Upper Middle Income")
plt.xlabel('Birth Rate')
plt.ylabel('Internet Users')
plt.legend()
plt.title('A Relationship between Birth Rate and Internet Users for Various Countries')
plt.show()








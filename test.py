from neural_network import NeuralNetwork
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=0)



y = y.reshape((y.shape[0], 1))

X = X.T
y = y.reshape((1, y.shape[0]))

# print("X dim " + str(X.shape))
# print("y dim " + str(y.shape))

# plt.scatter(X[0, :], X[1, :], c=y)
# plt.show()


network = NeuralNetwork([X.shape[0], 16, 16, 16, 1])

network.fit_(X, y, n_iter=2000, plot=True)
from neural_network import NeuralNetwork
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, n_features=4, centers=2, random_state=0)
y= y.reshape((y.shape[0], 1))

NeuralNetwork(X, [3, 3, 2])
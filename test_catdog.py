from utilities import *
import matplotlib.pyplot as plt
import numpy as np
from neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = load_data()

y_train =y_train.T
y_test = y_test.T

X_test = X_test.T
X_test_reshape = X_test.reshape(-1, X_test.shape[-1]) / X_train.max()

X_train = X_train.T
X_train_reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()

m_train = 300
m_test = 80
X_test_reshape = X_test_reshape[:, :m_test]
X_train_reshape = X_train_reshape[:, :m_train]
y_train = y_train[:, :m_train]
y_test = y_test[:, :m_test]

network = NeuralNetwork([X_train_reshape.shape[0], 32, 32, 32, 32, 32, 32, 1])

network.fit_(X_train_reshape, y_train, n_iter=8000, alpha=0.01, plot=True, y_test=y_test, X_test=X_test_reshape)

y_pred = network.predict_(X_test_reshape)

test_accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
print(test_accuracy)
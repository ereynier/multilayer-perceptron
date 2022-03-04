from neural_network_binary import NeuralNetworkBinary
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=10000, noise=0.1, factor=0.3, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_test = X_test.T
X_train = X_train.T


y_train = y_train.reshape((1, y_train.shape[0]))
y_test = y_test.reshape((1, y_test.shape[0]))


# print("X dim " + str(X.shape))
# print("y dim " + str(y.shape))



network = NeuralNetworkBinary([X_train.shape[0], 32, 32, 32, 1])

network.fit_(X_train, y_train, epoch=40, batch_size=100, plot=True)

y_pred = network.predict_(X_test)

plt.figure(figsize=(14,4))
plt.subplot(1, 2, 1)
plt.scatter(X_test[0, :], X_test[1, :], c=y_test)

plt.subplot(1, 2, 2)
plt.scatter(X_test[0, :], X_test[1, :], c=y_pred)


plt.show()
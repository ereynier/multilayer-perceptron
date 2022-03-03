from neural_network import NeuralNetwork
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification

X, y = make_circles(n_samples=10000, noise=0.1, factor=0.3, random_state=0)

y = np.expand_dims(y, 1)
y = np.concatenate((y, y), axis=1)
for i in range(len(y)):
    y[i][1] = (y[i][1] + 1) % 2

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_test = X_test.T
X_train = X_train.T


y_train = y_train.T
y_test = y_test.T

# print("X dim " + str(X.shape))
# print("y dim " + str(y.shape))


network = NeuralNetwork([X_train.shape[0], 32, 32, 32, y_train.shape[0]])

network.fit_(X_train, y_train, epoch=80, batch_size=500, plot=True)

y_pred = network.predict_(X_test)

plt.figure(figsize=(14,4))
plt.subplot(1, 2, 1)
plt.scatter(X_test[0, :], X_test[1, :], c=y_test[0])
plt.title("y_real")

plt.subplot(1, 2, 2)
plt.scatter(X_test[0, :], X_test[1, :], c=y_pred[0])
plt.title("y_pred")

plt.show()
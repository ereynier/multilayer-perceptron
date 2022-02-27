
import numpy as np

class Perceptron():
    def __init__(self, X):
        self.b = np.random.randn(1)
        self.W = np.random.randn(X.shape[1], 1)


    def model_(self, X, W, b):
        Z = X.dot(W) + b
        A = self.sigmoid_(Z)
        return (A)

    def gradient(self, A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return (dW, db)

    def update_(self, dW, db, W, b, alpha):
        W = W - alpha * dW
        b = b - alpha * db
        return (W, b)

    def fit_(self, X, y, alpha=0.1, n_iter=100):
        Loss = []
        for i in range(n_iter):
            A = self.model_(X, self.W, self.b)
            Loss.append(self.log_loss_(A, y))
            dW, db = self.gradient(A, X, y)
            self.W, self.b = self.update_(dW, db, self.W, self.b, alpha)
        return (self.W, self.b)

    def predict_(self, X, W, b):
        A = self.model_(X, W, b)
        return A >= 0.5

    def log_loss_(self, A, y):
        return (1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A)))

    def sigmoid_(self, z):
        return(1. / (1. + np.exp(-z)))
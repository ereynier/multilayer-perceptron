import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, layers):
        #first layer must be "X_train.shape[0]"
        #last layers can be "y_train.shape[0]"
        params = {}
        #-1 car 'layers' contient X_train.shape[0]
        self.n_layers = len(layers) - 1
        self.layers = layers
        for i in range(1, len(layers)):
            params["W" + str(i)] = np.random.randn(layers[i], layers[i - 1])
            params["b" + str(i)] = np.random.randn(layers[i], 1)
        self.params = params
        self.activations = {}

    def forward_propagation(self, X):
        A = X
        #set A0 to X for back_propagation A(i - 1)
        self.activations["A0"] = X
        for i in range(1, self.n_layers + 1):
            Z = self.params["W" + str(i)].dot(A) + self.params["b" + str(i)]
            self.activations["A" + str(i)] = self.sigmoid_(Z)
            self.activations["Z" + str(i)] = Z
            A = self.activations["A" + str(i)]
        return (self.activations)

    def back_propagation(self, X, y):
        m = y.shape[1]
        gradients = {}
        #déclaration de dZ pour éviter une erreur
        dZ = 0
        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                dZ = self.activations["A" + str(i)] - y
            else:
                dZ = np.dot(self.params["W" + str(i + 1)].T, dZ) * self.activations["A" + str(i)] * (1 - self.activations["A" + str(i)])
            gradients["dW" + str(i)] = 1 / m * dZ.dot(self.activations["A" + str(i - 1)].T)
            gradients["db" + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        self.gradients = gradients
        return (gradients)

    def update(self, alpha):
        for i in range(1, self.n_layers + 1):
            self.params["W" + str(i)] = self.params["W" + str(i)] - alpha * self.gradients["dW" + str(i)]
            self.params["b" + str(i)] = self.params["b" + str(i)] - alpha * self.gradients["db" + str(i)]
        return (self.params)

    def fit_(self, X, y, alpha=0.1, epoch=20, samples=200, plot=False, X_test=np.array([]), y_test=np.array([])):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        
        n_split = int(X.shape[1] / samples)
        if n_split == 0:
            n_split = 1
        print(f'Dataset split in {n_split}')
        X_split = np.array_split(X, n_split, axis=1)
        y_split = np.array_split(y, n_split, axis=1)

        for i in range(epoch):
            for j in range(n_split):
                self.forward_propagation(X_split[j])
                self.back_propagation(X_split[j], y_split[j])
                self.update(alpha)
                print(f'Split {j + 1}/{n_split}')
            
            #progress bar
            print(f'Epoch {i + 1}/{epoch}')

            #train_loss.append(self.cross_entropy_(self.activations["A" + str(self.n_layers)] ,y))
            y_pred = self.predict_(X)
            current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)

            if (np.any(X_test)) and (np.any(y_test)):
                #test_loss.append(self.cross_entropy_(self.activations["A" + str(self.n_layers)] ,y_test))
                y_test_pred = self.predict_(X_test)
                current_accuracy_test = accuracy_score(y_test.flatten(), y_test_pred.flatten())
                test_acc.append(current_accuracy_test)

        if plot == True:
            plt.figure(figsize=(14,4))

            plt.subplot(1, 2, 1)
            plt.plot(train_loss, label="train loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_acc, label="train accuracy")
            if (np.any(X_test) and np.any(y_test)):
                plt.plot(test_acc, label="test accuracy")
                print(test_acc[-1])
            plt.legend()
            plt.show()

            print(train_acc[-1])
        

        return (self.params)

    def predict_(self, X):
        self.forward_propagation(X)
        A = self.activations["A" + str(self.n_layers)]
        return A > 0.5

    def cross_entropy_(self, A, y):
        return -1 / self.layers[-1] * np.sum(y + np.log(A) + (1 - y) * np.log(1 - A))

    def log_loss_(self, A, y):
        return

    def softmax_(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def sigmoid_(self, z):
        return(1. / (1. + np.exp(-z)))
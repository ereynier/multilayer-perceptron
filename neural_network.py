import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from progressBar import progressBar

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
            #activation function
            if i == self.n_layers:
                #last neuron layers activation
                self.activations["A" + str(i)] = self.softmax_(Z)
            else:
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

    def fit_(self, X, y, alpha=0.1, epoch=20, batch_size=200, plot=False):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        
                

        for i in range(epoch):
            X_split, y_split, X_val, y_val, n_split = self.shuffle_split_(X, y, batch_size)
            #setup progress bar
            print(f'Epoch {i + 1}/{epoch}')
            bar = progressBar(range(n_split))
            bar.start()
            bar_msg = ""
            for j in range(n_split):
                self.forward_propagation(X_split[j])
                self.back_propagation(X_split[j], y_split[j])
                self.update(alpha)
                bar.update(j)


            y_pred = self.predict_(X_split[0])
            train_loss.append(self.cross_entropy_(self.activations["A" + str(self.n_layers)] ,y_split[0]))
            current_accuracy = accuracy_score(y_split[0].flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)
            
            # # VALIDATION SET
            y_val_pred = self.predict_(X_val)
            val_loss.append(self.cross_entropy_(self.activations["A" + str(self.n_layers)] ,y_val))
            current_accuracy_val = accuracy_score(y_val.flatten(), y_val_pred.flatten())
            val_acc.append(current_accuracy_val)

            print(f'Loss: {train_loss[-1]:.4f} - Val_loss: {val_loss[-1]:.4f} - Acc: {current_accuracy:.4f} - Val_acc: {current_accuracy_val:.4f} - Val size: {X_val.shape[1]}')

        if plot == True:
            plt.figure(figsize=(14,4))

            plt.subplot(1, 2, 1)
            plt.plot(train_loss, label="train loss")
            plt.plot(val_loss, label="val loss")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_acc, label="train accuracy")
            plt.plot(val_acc, label="val accuracy")

            plt.legend()
            plt.show()
        

        return (self.params)

    def predict_(self, X):
        self.forward_propagation(X)
        A = (self.activations["A" + str(self.n_layers)].T).copy()
        for k in range(len(A)):
            i = np.argmax(A[k], axis=0)
            A[k] = np.zeros(A[k].shape)
            A[k][i] = 1
        return A.T

    def cross_entropy_(self, A, y):
        return - (1 / self.layers[-1]) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

    def softmax_(self, z):
        s = np.zeros((1, z.shape[0]))
        for x in z.T:
            s = np.concatenate((s, np.array([np.exp(x) / np.sum(np.exp(x))])), axis=0)
        s = np.delete(s, 0, 0)
        return s.T

    def sigmoid_(self, z):
        return(1. / (1. + np.exp(-z)))

    def shuffle_split_(self, X, y, batch_size):
        #shuffle, split val set and batch
        seed = np.random.randint(0, 2147483647)
        X = X[:, np.random.RandomState(seed=seed).permutation(X.shape[1])]
        y = y[:, np.random.RandomState(seed=seed).permutation(y.shape[1])]

        X_train = np.delete(X, slice(int(-0.2 * X.shape[1]), None), axis=1)
        X_val = X[:, int(-0.2 * X.shape[1]):]
        y_train = np.delete(y, slice(int(-0.2 * y.shape[1]), None), axis=1)
        y_val = y[:, int(-0.2 * y.shape[1]):]
        
        n_split = int(X_train.shape[1] / batch_size)
        if n_split == 0:
            n_split = 1
        X_split = np.array_split(X_train, n_split, axis=1)
        y_split = np.array_split(y_train, n_split, axis=1)
        return (X_split, y_split, X_val, y_val, n_split)
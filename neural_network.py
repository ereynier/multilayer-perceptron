import numpy as np
from perceptron import Perceptron

class NeuralNetwork():
    def __init__(self, X, n_neuron):
        layers = []
        for i in range(len(n_neuron)):
            layers.append([])
            for n in range(n_neuron[i]):
                layers[i].append(Perceptron(X))
        self.layers = layers
        print(layers[0][0].W)
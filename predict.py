import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import yaml
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

def renameCol(data):
    data.rename(columns={0 : "ID", 
                         1 : "Diagnosis",
                         2 : "radius",
                         3 : "texture",
                         4 : "perimeter",
                         5 : "area",
                         6 : "smoothness",
                         7 : "compactness",
                         8 : "concavity",
                         9 : "concave points",
                         10 : "symmetry",
                         11 : "fractal"}, inplace=True)
    data.set_index('ID', inplace=True)
    suffix = [i + "_SE" for i in list(data.columns[1:11])]
    data.rename(columns=dict(zip(data.columns[11:22], suffix)), inplace=True)
    suffix = [i + "_Worst" for i in list(data.columns[1:11])]
    data.rename(columns=dict(zip(data.columns[21:33], suffix)), inplace=True)
    return (data)

def selectData(df):
    with open('config.yml', "r") as f:
        config = yaml.safe_load(f)
    #USE CONFIG FILE
    X = df[config["features"]]
    return (X)

def normalize(df):
    normalized_df = (df - df.mean()) / df.std()
    return (normalized_df)

def main():
    try:
        data = pd.read_csv(sys.argv[1], header=None)
    except:
        print("Can't read data.csv")
        return
    
    try:
        filename = sys.argv[2]
    except:
        print("please enter a filename for weights and layers")
        return

    data=renameCol(data)
    df = data.copy()

    enc = OneHotEncoder(sparse=False)

    y = enc.fit_transform(df[["Diagnosis"]].values)

    X = selectData(df)
    #X = df.drop("Diagnosis", axis=1)

    X = normalize(X)

    X = X.to_numpy()

    #transpose X y
    X = X.T
    y = y.T

    try:
        f = open(filename + ".layers", "r")
        layers = f.read()
        layers = str(layers)
        f.close()
    except Exception as e:
        print("Error while reading layers : " + e)
    layers = list(map(int, layers.split(', ')))
    layers[0] = X.shape[0]

    network = NeuralNetwork(layers)
    

    network.load_(filename + ".npy")

    y_pred = network.predict_(X)

    print(f'Loss: {network.cross_entropy_(network.activations["A" + str(network.n_layers)] ,y):.4f}')

    #y_test proportion de bonne réponses
    y_pred = enc.inverse_transform(y_pred.T)
    y = enc.inverse_transform(y.T)
    c = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y[i]:
            c += 1
    print(f'Good prediction: {(c / len(y_pred))*100:.2f}%')
    print(f'Accuracy: {accuracy_score(y.flatten(), y_pred.flatten()):.4f}')
    print(f'F1 Score: {f1_score(y.flatten(), y_pred.flatten(), pos_label="M"):.4f}')
    #print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
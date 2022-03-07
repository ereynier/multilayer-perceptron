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
    data=renameCol(data)
    df = data.copy()

    enc = OneHotEncoder(sparse=False)

    y = enc.fit_transform(df[["Diagnosis"]].values)

    X = selectData(df)
    #X = df.drop("Diagnosis", axis=1)

    X = normalize(X)


    X = X.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #transpose X y
    X_test = X_test.T
    X_train = X_train.T
    y_train = y_train.T
    y_test = y_test.T

    # layers = [4] * 50
    # layers.append(y_train.shape[0])
    # layers.insert(0, X_train.shape[0])

    network = NeuralNetwork([X_train.shape[0], 32, 32, 32, y_train.shape[0]])
    
    network.fit_(X_train, y_train, epoch=80, batch_size=500, plot=True, early_stopping=0)

    y_pred = network.predict_(X_test)

    print(f'Loss: {network.cross_entropy_(network.activations["A" + str(network.n_layers)] ,y_pred):.4f}')

    #y_test proportion de bonne réponses
    y_pred = enc.inverse_transform(y_pred.T)
    y_test = enc.inverse_transform(y_test.T)
    c = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            c += 1
    print(f'Good prediction: {(c / len(y_pred))*100:.2f}%')
    print(f'Accuracy: {accuracy_score(y_test.flatten(), y_pred.flatten())}')


if __name__ == "__main__":
    main()
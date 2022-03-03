import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

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

def plotData(df):
    print(df.describe())
    plt.figure(figsize=(12,20))
    sns.heatmap(df.corr())
    plt.show()

    # df = df[["Diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal"]]
    # sns.pairplot(df, hue="Diagnosis")
    # plt.show()

    for col in df.select_dtypes(float):
        sns.displot(x=df[col], hue=df["Diagnosis"], kde=True)
        plt.show()

def selectData(df):
    #USE CONFIG FILE
    X = df.drop("Diagnosis", axis=1)
    return (X)

def normalize(df):
    return (df)

def main():
    try:
        data = pd.read_csv("data/data.csv", header=None)
    except:
        print("Can't read data.csv")
        return
    data=renameCol(data)
    print(data.head)

    #get target 'Diagnosis' as 'y'
    df = data.copy()

    #IF DISPLAY = TRUE
    #plotData(df)

    enc = OneHotEncoder(sparse=False)
    y = enc.fit_transform(df[["Diagnosis"]].values)

    #print(enc.inverse_transform(y))

    X = selectData(df)

    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    


if __name__ == "__main__":
    main()
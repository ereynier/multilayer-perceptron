import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import yaml
import sys

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
    fig = plt.figure(figsize=(30,30))
    sns.heatmap(df.corr())
    plt.savefig("fig/corr_heatmap.png")

    print(df["Diagnosis"].value_counts(normalize=True))

    i = 1
    fig = plt.figure(figsize=(30,30))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for col in df.select_dtypes(float):
        ax = fig.add_subplot(6, 6, i)
        i += 1
        sns.histplot(x=df[col], hue=df["Diagnosis"], kde=True, ax=ax)
    plt.savefig(f'fig/dist.png')


def main():
    try:
        data = pd.read_csv(sys.argv[1], header=None)
    except:
        print("Can't read data.csv")
        return
    data=renameCol(data)
    df = data.copy()

    plotData(df)


if __name__ == "__main__":
    main()
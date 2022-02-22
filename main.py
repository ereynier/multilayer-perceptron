import pandas as pd


def main():
    try:
        data = pd.read_csv("data/data.csv", header=None)
        print(data.head())
    except:
        print("Can't read data.csv")
        return


if __name__ == "__main__":
    main()
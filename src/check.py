import pandas as pd
def print_shape(X,Y):
    print(X.shape[0])
    print("\n")
    print(X.shape[1])
    print("\n")
    print(Y.shape[0])
    print("\n")
    print(Y.shape[1])

def cut_columns(df):
    df = df.iloc[:,0:32]
    df = df.fillna(0)
    return df


if __name__ == '__main__':
    X = pd.read_csv("train_X.csv")
    Y = pd.read_csv("train_Y.csv")
    X = cut_columns(X)
    Y = cut_columns(Y)
    print(X,Y)
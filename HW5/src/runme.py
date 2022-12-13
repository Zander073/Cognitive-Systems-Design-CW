import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "../dat/reg_data.csv"

# Gets expected value of variable
def get_ev(df, v):
    return df.mean()[v]

# Gets variance of a variable
def get_variance(df, v):
    return df.var()[v]

# Print ev, cov, and corr
def get_expected_values(df):
    print("Expected values for: ", df.columns)
    ev = df.mean()
    print(ev, "\n")
    return ev
def get_covariance(df):
    print("Covariance for: ", df.columns)
    cov = df.cov()
    print(cov, "\n")
    return cov
def get_correlation(df):
    print("Correlation for: ", df.columns)
    corr = df.corr()
    print(corr, "\n")
    return corr

if __name__ == "__main__":
    df = pd.read_csv(data_path)

    # Number 1:
    get_expected_values(df)
    print("Variance:")
    for column in df:
        print(column, " ", get_variance(df, column))
    print()
    get_covariance(df)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Number 2:
    # X -> Y
    x = 'X'
    y = 'Y'
    cov = df.cov().at[x, y]
    var = get_variance(df, x)
    ev_x = get_ev(df, x)
    ev_y = get_ev(df, y)
    beta = (cov / var)
    alpha = ev_x - (ev_y * beta)
    pairs_one_x = []
    pairs_one_y = []


    for index, rows in df.iterrows():
        value = alpha + beta*rows[x]
        pairs_one_x.append(rows[x])
        pairs_one_y.append(value)
    
    y = 'Z'
    cov = df.cov().at[x, y]
    ev_y = get_ev(df, y)
    beta = (cov / var)
    alpha = ev_x - (ev_y * beta)
    pairs_two_y = []

    for index, rows in df.iterrows():
        value = alpha + beta*rows[x]
        pairs_two_y.append(value)

    plt.scatter(pairs_one_x, pairs_one_y, label = "X -> Y", color = "red")
    plt.scatter(pairs_one_x, pairs_two_y, label = "X -> Z", color = "blue")
    plt.xlabel("X values")
    plt.ylabel("LR Outputs")
    plt.legend()
    plt.show()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Number 3:

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Number 4:

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Number 5:
    

    
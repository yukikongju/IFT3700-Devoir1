#!/env/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def group_4_pixels(x):
    variables_size = (2, 2)
    n_variables = 784 // np.prod(variables_size)
    n_outcomes = 2**np.prod(variables_size)
    outcomes = np.zeros((len(x), n_variables), dtype=int)

    if isinstance(variables_size, int):
        reshaped_x = x.reshape(-1, n_variables, variables_size)
    else: #variables_size is a 2-tuples
        a, b = variables_size 
        reshaped_x = x.reshape(-1, 28//a, a, 28//b, b).transpose(0, 2, 1, 4, 3).reshape(-1, 784//(a*b), a*b)
        variables_size = a*b
    for n in range(variables_size):
        outcomes += reshaped_x[:,:,variables_size-n-1] * 2**n
    return outcomes



def main():
    # load dataset
    mnist_test = pd.read_csv("mnist/mnist_test.csv")
    mnist_train = pd.read_csv("mnist/mnist_train.csv")

    # separate target(Y) from features(X)
    y_train = mnist_train['label']
    x_train = mnist_train.drop(columns=['label'])

    x_test = mnist_test.drop(columns=['label'])
    y_test = mnist_test['label']

    # normalize pixels on features
    normalize_pixels(x_train)
    normalize_pixels(x_test)

    #  group pixels by 4 x=(x0, x1, x28, x29)
    compressed_x_train = group_4_pixels(x_train.to_numpy())
    compressed_x_test = group_4_pixels(x_test.to_numpy())

    # create clean dataset
    column_names = [f"{i}x{j}" for j in range(1,15) for i in range(1,15)]
    mnist_train_clean = pd.DataFrame(compressed_x_train, columns= column_names)
    mnist_test_clean = pd.DataFrame(compressed_x_test, columns= column_names)
    mnist_train_clean['label'] = y_train
    mnist_test_clean['label'] = y_test

    # save file
    mnist_test_clean.to_csv('mnist/mnist_test_clean.csv')
    mnist_train_clean.to_csv('mnist/mnist_train_clean.csv')


def is_activated(pixel):
    if pixel == 0: 
        return False
    return True

def normalize_pixels(df):
    for column in df.columns:
        df[column] = df[column].apply(lambda x: 1 if is_activated(x) else 0)

def plot_img(): # TODO
    pass

if __name__ == "__main__":
    main()




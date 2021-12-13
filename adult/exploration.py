#!/env/bin/python 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # importer dataset
    df = pd.read_csv('adult/adult.csv')
    df_clean = pd.read_csv('adult/adult_clean.csv')

    # histogrammes
    df_nums = df.select_dtypes(include = ['int64', 'float64'])
    df_nums.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

    # correlation
    sns.pairplot(df)
    sns.heatmap(df.corr(), annot= True, cmap = "Greens")

    # countplot
    sns.countplot(df["income"])
    print("percentage population making under 50K: ", round(len(df[df["income"] == "<=50K"]) / len(df)*100, 3), "%")
    print("percentage population making under 50K: ", round(len(df[df["income"] == ">50K"]) / len(df)*100, 3), "%")

    # countplot des incomes par catégorie
    categories = ['education', 'native-country', 'age', 'race', 'occupation', 
            'workclass', 'relationship', 'marital-status', 'hours-per-week']
    for categorie in categories:
        count_plot(df, categorie, 'income')
        print_dataframe_class_ratio(df, categorie, 'income')

def count_plot(df, feature, target):
    sns.countplot(y=df['education'], hue=df['income'])

def dist_plot(df, feature, target):
    sns.displot(df, x=df[feature], hue=df[target], kind='kde')

def cat_plot(df, feature, target):
    pass


def print_dataframe_class_ratio(df, feature, target):
    """ Imprime le tableau du pourcentage de classes par feature """
    ratios = df.groupby(by=[feature])[target].value_counts(normalize=True)
    ratios = ratios.mul(100).rename('Percent').reset_index()
    print(ratios)


if __name__ == "__main__":
    main()


#!/env/bin/python

import numpy as np 
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier 

from algorithms.isomap import IsomapCustom
from algorithms.kmedoids import KMedoidsCustom
from algorithms.knn import KNNCustom
from algorithms.partition import PartitionBinaireCustom
from algorithms.pcoa import PCoACustom

from adult.dissimilarity import get_adult_dissimilarity_matrix
from mnist.dissimilarity import get_mnist_dissimilarity_matrix

def get_training_testing_validation(df, target):
    """ retourne 3 ensembles de données: entrainement, test et validation 
    selon on pourcentage de données"""
    # separate features and target
    y = df[target]
    x = df.drop(columns=[target])

    # separate testing, training and validation set
    x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=2, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.3, random_state=2, stratify=y_train)

    return x_train, x_val, x_test, y_train, y_val, y_test

def get_mnist_clustering_prediction_accuracy(df, clust_pred, true_y, num_clusters):
    """ Compare cluster prediction for MNIST: on veut associer chaque cluster 
    à la classe du target le plus populaire au sein de ce cluster"""

    # add label and cluster to df
    df['label'] = true_y
    df['cluster'] = clust_pred
    
    # compter le nombre de lignes associés à la classe i: num_clustersx10
    class_counts = np.zeros((num_clusters,10), dtype=int) 
    for _, row in df.iterrows():
        cluster = row['cluster']
        label = row['label']
        class_counts[cluster, label] += 1
    #  print(class_counts)

    # associate cluster to class
    clust_dict = {}
    for cluster in range(num_clusters):
        most_pop_class = 0
        max_count = 0
        for i, val in enumerate(class_counts[cluster]):
            if val > max_count:
                most_pop_class = i
                max_count = val
        clust_dict[cluster] = most_pop_class

    # calculate prediction accuracy
    df['cluster_pred'] = df['cluster'].apply(lambda x: clust_dict[x])
    accurate_pred_count = 0
    for _, row in df.iterrows():
        if row['label'] == row['cluster_pred']:
            accurate_pred_count += 1
    return accurate_pred_count/len(df)


def get_adult_clustering_prediction_accuracy(df, clust_pred, true_y, num_clusters,
        threshold=0.5):
    """ Compare cluster prediction with '<50K' and '>=50K' """
    # calculate percentage of each class by cluster
    df['income'] = true_y
    df['cluster'] = clust_pred

    # compter le nombre de lignes associés à la classe first row:'<=50K' et 2nd: '>50K'
    class_counts = np.zeros((num_clusters,2), dtype=int) 
    for _, row in df.iterrows():
        cluster = row['cluster']
        if row['income'] == "<=50K": col = 0
        else: col = 1
        class_counts[cluster, col] += 1

    # associate cluster to class
    clust_dict = {}
    for cluster in range(num_clusters):
        most_pop_class = 0
        max_count = 0
        if class_counts[cluster, 0] > class_counts[cluster, 1]: # if num of 
            clust_dict[cluster] = '<=50K'
        else: 
            clust_dict[cluster] = '>50K'

    # calculate accurate prediction
    df['cluster_pred'] = df['cluster'].apply(lambda x: clust_dict[x])
    accurate_pred_count = 0
    for _, row in df.iterrows():
        if row['income'] == row['cluster_pred']:
            accurate_pred_count += 1
    return accurate_pred_count/len(df)

def get_random_values_in_range(start, end, n):
    """ return list of n unique value within range """
    return random.sample(range(start, end), n)


def get_dimension_reduction_classification_score(transformed_features, y_true,
        n_neighbors): 
    """ Tester la réduction de dimensionalité en entrainant les données 
    réduites sur un algo de classification ie KNN """

    # get indices to use for train and test dataset
    len_train = len(transformed_features) // 3
    len_test = len(transformed_features) // 6
    train_index = get_random_values_in_range(1, len(transformed_features), len_train)
    test_index = get_random_values_in_range(1, len(transformed_features), len_test)

    # split dataset
    x_train = np.array([transformed_features[i] for i in train_index])
    x_test = np.array([transformed_features[i] for i in test_index])

    y_train = [np.array(y_true)[i] for i in train_index]
    y_test = [np.array(y_true)[i] for i in test_index]

    # classifier
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, 
            metric = 'euclidean')  
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)


def ADULT_main():
    """ importer, séparer et rouler les algorithmes obligatoires sur le dataset ADULT """
    # import cleaned up dataset
    adult = pd.read_csv("adult/adult_clean.csv")

    # split dataset into training, testing and validation
    x_train, x_val, x_test, y_train, y_val, y_test = get_training_testing_validation(adult, 'income')

    # sample randomly dataset to reduce computation cost (on choisi 100 sample)
    train_index = get_random_values_in_range(1, len(x_train), 100)
    test_index = get_random_values_in_range(1, len(x_test), 100)
    val_index = get_random_values_in_range(1, len(x_val), 100)

    # split dataset with indices selected
    x_train = x_train.iloc[train_index]
    x_test = x_test.iloc[test_index]
    x_val = x_val.iloc[val_index]
    y_train = y_train.iloc[train_index]
    y_test = y_test.iloc[test_index]
    y_val = y_val.iloc[val_index]

    # calculate dissimilarity matrix for x_train and save 
    dissimilarity_matrix_train = get_adult_dissimilarity_matrix(x_train)
    dissimilarity_matrix_test = get_adult_dissimilarity_matrix(x_test)
    dissimilarity_matrix_val = get_adult_dissimilarity_matrix(x_val)

    # test KNN - worked! (classification)
    knn = KNNCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, y_test, dissimilarity_matrix_val, y_val)

    # test Isomap 
    isomap = IsomapCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, 
            y_test, dissimilarity_matrix_val, y_val)
    isomap_classification_accuracy = get_dimension_reduction_classification_score(
            isomap.transformed_features, y_test, isomap.best_n_neighbors)
    print(f"Classification accuracy pour Isomap sur ADULT: {isomap_classification_accuracy}")

    # test PCoA - worked! (dimension reduction)
    pcoa = PCoACustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, y_test, dissimilarity_matrix_val, y_val)
    pcoa_classification_accuracy = get_dimension_reduction_classification_score(
            pcoa.transformed_features, y_test.to_numpy(), pcoa.best_n_components)
    print(f"Classification Accuracy pour PCoA sur ADULT: {pcoa_classification_accuracy}")

    # test KMedoids - 
    kmedoids = KMedoidsCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, y_test, dissimilarity_matrix_val, y_val)
    kmedoids_accuracy = get_adult_clustering_prediction_accuracy(x_test,
            kmedoids.cluster_prediction, y_test, kmedoids.best_n_clusters)
    print(f"Accuracy pour KMedoids sur ADULT: {kmedoids_accuracy}")

    # test Partitionnement Binaire (clustering)
    partition = PartitionBinaireCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, y_test, dissimilarity_matrix_val, y_val)
    partition_accuracy = get_adult_clustering_prediction_accuracy(x_test, 
            partition.cluster_prediction, y_test, partition.best_n_clusters)
    print(f"Accuracy pour Partitionnement Binaire sur ADULT: {partition_accuracy}")

def MNIST_main(): 
    """ importer, séparer et rouler les algorithmes obligatoires sur le dataset MNIST """
    # import cleaned up dataset
    mnist = pd.read_csv("mnist/mnist_train_clean.csv")

    # split dataset into training, testing and validation
    x_train, x_val, x_test, y_train, y_val, y_test = get_training_testing_validation(mnist, 'label')

    # select subset to reduce computation cost
    # sample randomly dataset to reduce computation cost (on choisi 100 sample)
    train_index = get_random_values_in_range(1, len(x_train), 100)
    test_index = get_random_values_in_range(1, len(x_test), 100)
    val_index = get_random_values_in_range(1, len(x_val), 100)

    # split dataset with indices selected
    x_train = x_train.iloc[train_index]
    x_test = x_test.iloc[test_index]
    x_val = x_val.iloc[val_index]
    y_train = y_train.iloc[train_index]
    y_test = y_test.iloc[test_index]
    y_val = y_val.iloc[val_index]

    # calculate dissimilarity matrix for x_train and save 
    dissimilarity_matrix_train = get_mnist_dissimilarity_matrix(x_train)
    dissimilarity_matrix_test = get_mnist_dissimilarity_matrix(x_test)
    dissimilarity_matrix_val = get_mnist_dissimilarity_matrix(x_val)

    # test KNN Classifier
    knn = KNNCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, 
            y_test, dissimilarity_matrix_val, y_val)

    # test Isomap 
    isomap = IsomapCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, 
            y_test, dissimilarity_matrix_val, y_val)
    isomap_classification_accuracy = get_dimension_reduction_classification_score(
            isomap.transformed_features, y_test, isomap.best_n_neighbors)
    print(f"Isomap Accuracy sur MNIST: {isomap_classification_accuracy}")

    # test PCoA - didnt work! (dimension reduction)
    #  pcoa = PCoACustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, y_test, dissimilarity_matrix_val, y_val)
    #  pcoa_classification_accuracy = get_dimension_reduction_classification_score(
            #  pcoa.transformed_features, y_test, pcoa.best_n_components)
    #  print(f"Classification Accuracy pour PCoA sur MNIST: {pcoa_classification_accuracy}")

    # test KMedoids - 
    kmedoids = KMedoidsCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, y_test, dissimilarity_matrix_val, y_val)
    print(kmedoids.cluster_prediction)
    kmedoids_accuracy = get_mnist_clustering_prediction_accuracy(x_test,
            kmedoids.cluster_prediction, y_test, kmedoids.best_n_clusters)
    print(f"Accuracy pour KMedoids sur MNIST: {kmedoids_accuracy}")

    # test Partitionnement Binaire (clustering)
    partition = PartitionBinaireCustom(dissimilarity_matrix_train, y_train, dissimilarity_matrix_test, y_test, dissimilarity_matrix_val, y_val)
    partition_accuracy = get_mnist_clustering_prediction_accuracy(x_test, 
            partition.cluster_prediction, y_test, partition.best_n_clusters)
    print(f"Accuracy pour Partitionnement Binaire sur MNIST: {partition_accuracy}")


def main():
    # set seed for reproducibility
    np.random.seed(420)

    # tester les algorithmes sur les datasets ADULT et MNIST
    ADULT_main()
    MNIST_main()


if __name__ == "__main__":
    main()



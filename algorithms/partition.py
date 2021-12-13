#!/env/bin/python

import numpy as np
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

from algorithms._base import ClusteringAlgorithm

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering


class PartitionBinaireCustom(ClusteringAlgorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)

        self.best_n_clusters = self.get_best_hyperparameters()

        self.cluster_prediction = self.get_cluster_prediction()
        self.print_results()

    def get_best_hyperparameters(self, start_n=2, end_n=10):
        best_silhouette_score = 0
        best_n_clusters = start_n
        for num_clusters in range(start_n, end_n):
            agglomerative_clustering = AgglomerativeClustering(
                    n_clusters=num_clusters, affinity='precomputed', 
                    linkage='average')
            agglomerative_clustering.fit(self.x_train)
            labels = self.agglomerative_clustering_predict(agglomerative_clustering, 
                    self.x_train)

            silhouette_avg = silhouette_score(self.x_train, labels)
            if silhouette_avg > best_silhouette_score:
                best_n_clusters = num_clusters
                best_silhouette_score = silhouette_avg
        print(f"Best Silhouette Score: {best_silhouette_score}")

        return best_n_clusters

    def agglomerative_clustering_predict(self, agglomerative_clustering, x):
        average_dissimilarity = list()
        for i in range(agglomerative_clustering.n_clusters):
            ith_clusters_dissimilarity = x[:, np.where(agglomerative_clustering.labels_==i)[0]]
            average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
        return np.argmin(np.stack(average_dissimilarity), axis=0)

    def plot_clustering(self):
        dendogram = shc.dendrogram(shc.linkage(self.x_train, method='ward'))
        plt.figure(figsize=(10,7))
        plt.title("Dendrograms")
        plt.show()

    def get_cluster_prediction(self): 
        agglomerative_clustering = AgglomerativeClustering(
                n_clusters=self.best_n_clusters, affinity='precomputed', 
                linkage='average')
        agglomerative_clustering.fit(self.x_train)
        return self.agglomerative_clustering_predict(agglomerative_clustering, 
                self.x_test)

    def print_results(self):
        print(f"Le meilleur nombre de cluster pour partition binaire est: {self.best_n_clusters}")


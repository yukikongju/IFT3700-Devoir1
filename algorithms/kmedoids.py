#!/env/bin/python

from algorithms._base import ClusteringAlgorithm

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, silhouette_samples

class KMedoidsCustom(ClusteringAlgorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)

        self.best_n_clusters = self.get_best_n_clusters()
        self.cluster_prediction = self.get_cluster_prediction()
        self.print_results()

    def get_best_n_clusters(self, start_n=2, end_n=20): 
        best_silhouette_score = 0
        best_n_clusters = start_n
        for num_clusters in range(start_n, end_n):
            # create kmedoid model
            kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', 
                    random_state=420)
            #  kmedoids.fit(self.x_train)
            #  cluster_labels = kmedoids.predict(self.x_val)
            cluster_labels = kmedoids.fit_predict(self.x_val)
            #  centers = kmedoids.cluster_centers_

            # compute silouhette score
            silhouette_avg = silhouette_score(self.x_val, labels=cluster_labels)
            sample_silhouette_values = silhouette_samples(self.x_val,
                    labels=cluster_labels)
            print(f"Pour {num_clusters} clusters, le score silouhette est: {silhouette_avg}")

            # update best silouhette score and best n_clusters
            if best_silhouette_score < silhouette_avg:
                best_silhouette_score = silhouette_avg
                best_n_clusters = num_clusters

        print(f"Le meilleur score silhouette est: {round(best_silhouette_score*100,3)}")

        return best_n_clusters 

    def get_cluster_prediction(self):
        """ Return transformed features """
        kmedoids = KMedoids(n_clusters=self.best_n_clusters, metric='precomputed', 
                random_state=420)
        kmedoids.fit(self.x_train)
        return kmedoids.predict(self.x_test)

    def plot_clustering(self):
        pass


    def print_results(self):
        print(f"Le meilleur nombre de cluster pour KMedoids est: {self.best_n_clusters}")



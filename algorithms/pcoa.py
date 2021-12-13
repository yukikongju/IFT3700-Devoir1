#!/env/bin/python

from algorithms._base import ClusteringAlgorithm

from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class PCoACustom(ClusteringAlgorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)

        self.best_n_components = self.get_best_hyperparameters()
        self.transformed_features = self.get_transformed_features()

        print(f"The best components is: {self.best_n_components}")
        #  print(f"Classification Score for PCoA and KNN: {self.classification_score}")


    def get_best_hyperparameters(self, start_c=2, end_c=10): 
        best_classifier_accuracy = 0
        best_n_components = start_c
        for n_components in range(start_c, end_c):
            # get features reduction from pcoa
            pcoa = KernelPCA(n_components=n_components, kernel='precomputed', 
                    random_state=420)
            transformed_features = pcoa.fit_transform(-.5*self.x_val**2) 
            
            # test transformed features on classifier
            x_transformed_train, x_transformed_test, y_transformed_train, y_transformed_test = train_test_split(
                    transformed_features, self.y_train, test_size=0.3, random_state=2)
            classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean') # n_neighbors est arbitraire
            classifier.fit(x_transformed_train, y_transformed_train)
            y_predict = classifier.predict(x_transformed_test)
            classifier_accuracy = accuracy_score(y_predict, y_transformed_test)*100
            
            # update hyperparameters
            if classifier_accuracy > best_classifier_accuracy:
                best_classifier_accuracy = classifier_accuracy
                best_n_components = n_components

        return best_n_components

    def get_transformed_features(self):
        pcoa = KernelPCA(n_components=self.best_n_components,
                kernel='precomputed', random_state=420)
        transformed_features_train = pcoa.fit_transform(self.x_train)
        return pcoa.fit_transform(self.x_test)


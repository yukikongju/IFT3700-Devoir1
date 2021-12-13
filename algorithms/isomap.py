#!/env/bin/python

from algorithms._base import DimensionReductionAlgorithm

from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class IsomapCustom(DimensionReductionAlgorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)
        
        self.best_n_components, self.best_n_neighbors = self.get_best_hyperparameters()
        self.transformed_features = self.get_transformed_features()
        self.print_results()


    def get_best_hyperparameters(self, start_n=2, end_n=15, start_c=2, end_c=15): 
        best_classifier_accuracy = 0
        best_n_components = start_c
        best_n_neighbors = start_n
        
        for n_components in range(start_c, end_c):
            for n_neighbors in range(start_n, end_n):
                # isomap model
                isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors,
                        metric = 'precomputed')
                transformed_features = isomap.fit_transform(self.x_train)   # ecq on veut x_train ou x_val?

                # train and test classfier with transformed features
                x_transformed_train, x_transformed_test, y_transformed_train, y_transformed_test = train_test_split(
                    transformed_features, self.y_train, test_size=0.3, random_state=2)

                # test with classifier
                classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean') # add justification
                classifier.fit(x_transformed_train, y_transformed_train)
                y_predict = classifier.predict(x_transformed_test)
                classifier_accuracy = accuracy_score(y_predict, y_transformed_test)
                
                # update hyperparameters
                if classifier_accuracy > best_classifier_accuracy:
                    best_classifier_accuracy = classifier_accuracy
                    best_n_components = n_components
                    best_n_neighbors = n_neighbors

        print(f"Meilleur training accuracy pour Isomap: {best_classifier_accuracy}")

        return best_n_components, best_n_neighbors


    def get_transformed_features(self):
        isomap = Isomap(n_components=self.best_n_components, 
                n_neighbors=self.best_n_neighbors, metric = 'precomputed')
        isomap.fit(self.x_train)
        return isomap.transform(self.x_test)

    def print_results(self):
        print(f"Meilleur n_components pour Isomap: {self.best_n_components} ")
        print(f"Meilleur n_neighbors pour Isomap: {self.best_n_neighbors}")


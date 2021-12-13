#!/env/bin/python

from algorithms._base import ClassificationAlgorithm

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

class KNNCustom(ClassificationAlgorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)

        self.best_k = self.get_best_hyperparameters()
        self.accuracy_test = self.get_test_accuracy()
        self.print_results()
        
    def get_best_hyperparameters(self, start_k = 1, end_k = 20):
        """ Find the best K with training and validation set """
        parameters = {"n_neighbors": range(start_k, end_k)}
        grid = GridSearchCV(KNeighborsClassifier(metric='precomputed', 
            algorithm='brute'), parameters, scoring='accuracy')
        grid.fit(self.x_train, self.y_train) 
        val_pred_grids = grid.predict(self.x_val)
        best_k = grid.best_params_["n_neighbors"] 
        train_accuracy = grid.best_score_ * 100

        print(f"The accuracy with the training dataset is: {train_accuracy}")

        return best_k

    def get_test_accuracy(self):
        knn = KNeighborsClassifier(n_neighbors=self.best_k, metric='precomputed', 
                algorithm='brute')
        knn.fit(self.x_train, self.y_train)
        y_predict = knn.predict(self.x_test)

        return accuracy_score(y_predict, self.y_test)*100

    def plot_confusion_matrix(self):
        pass

    def print_results(self):
        print(f"The K minimizing the error is: {self.best_k}")
        print(f"The accuracy with the test dataset is: {self.accuracy_test}")


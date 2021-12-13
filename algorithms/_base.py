#!/env/bin/python

""" Classe mère pour les algorithmes """


class Algorithm(object):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

    def get_best_hyperparameters(self):
        """ trouver les meilleurs hyperparamètres à l'aide des données d'entrainements 
        et de validation """
        pass

    def print_results(self):
        """ Imprimer les résultats de l'algorithme: meilleurs hyperparamètres, 
        accuracy score, ..."""
        pass


class ClassificationAlgorithm(Algorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)


    def get_test_accuracy(self):
        """ tester le accuracy de l'algorithme de classification sur 
        l'ensemble test après avoir trouvé les meilleurs hyperparamètres """
        pass
        
class ClusteringAlgorithm(Algorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)

    def get_cluster_prediction(self):
        """ retourne le cluster dans lequel l'algorithme croit que l'individu 
        fait partie """
        pass

    def plot_clustering(self):
        pass


class DimensionReductionAlgorithm(Algorithm):

    def __init__(self, x_train, y_train, x_test, y_test, x_val, y_val):
        super().__init__(x_train, y_train, x_test, y_test, x_val, y_val)

    def get_transformed_features(self):
        """ retourne un numpy array avec les données réduites """
        pass


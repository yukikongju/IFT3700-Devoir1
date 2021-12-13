#!/env/bin/python 

import pandas as pd
import numpy as np
import sklearn as skl

from itertools import combinations
from adult.constants import DissimilarityMatrix, DissimilarityMatrixIndex


#######################################################################
#                           Meta functions                            #
#######################################################################


def get_difference_value(dissimilarity_matrix, dissimilarity_dict_index, value1, value2):
    """ Get dissimilarity score between two individuals based on features value """
    i = dissimilarity_dict_index[value1]
    j = dissimilarity_dict_index[value2]
    return dissimilarity_matrix[i][j]

def is_different(value1, value2):
    if value1 == value2:
        return 0
    return 1

#######################################################################
#                   Calculate Dissimilarity Values                    #
#######################################################################

""" on veut trouver le score de dissimilarité de 2 individu en regardant 
la matrice de dissimilarité dans constants.py"""

def get_workclass_dissimilarity(workclass1, workclass2):
    return get_difference_value(DissimilarityMatrix.WORKCLASS, 
            DissimilarityMatrixIndex.WORKCLASS, workclass1, workclass2)

def get_education_dissimilarity(education1, education2):
    return get_difference_value(DissimilarityMatrix.EDUCATION, 
            DissimilarityMatrixIndex.EDUCATION, education1, education2)

def get_country_dissimilarity(country1, country2):
    return get_difference_value(DissimilarityMatrix.COUNTRY, 
            DissimilarityMatrixIndex.COUNTRY, country1, country2)

def get_age_dissimilarity(age1, age2):
    return get_difference_value(DissimilarityMatrix.AGE, 
            DissimilarityMatrixIndex.AGE, age1, age2)

def get_occupation_dissimilarity(occupation1, occupation2):
    return get_difference_value(DissimilarityMatrix.OCCUPATION, 
            DissimilarityMatrixIndex.OCCUPATION, occupation1, occupation2)

def get_hours_dissimilarity(hours1, hours2):
    return get_difference_value(DissimilarityMatrix.HOURS, 
            DissimilarityMatrixIndex.HOURS, hours1, hours2)

#######################################################################
#                   Calculate Dissimilarity Matrix                    #
#######################################################################



def adult_dissimilarity(row1, row2, *args, **kwargs):
    """ Trouver le score de dissimilarité entre 2 individus en additionnant 
    la différence entre chaque feature, pondérée par l'importance de celles-ci"""
    diff_age = get_age_dissimilarity(row1['age_label'], row2['age_label'])
    diff_workclass = get_workclass_dissimilarity(row1['workclass'], row2['workclass'])
    diff_education = get_education_dissimilarity(row1['education'], row2['education'])
    diff_marital_status = is_different(row1['has_partner'], row2['has_partner'])
    diff_occupation = get_occupation_dissimilarity(row1['occupation'], row1['occupation'])
    diff_race = is_different(row1['race'], row1['race'])
    diff_capital = is_different(row1['is_investing'], row2['is_investing'])
    diff_hours_per_week = get_hours_dissimilarity(row1['hours_per_week_label'],
            row2['hours_per_week_label'])
    diff_native_country = get_country_dissimilarity(row1['native-country'],
            row2['native-country'])
    return 2*diff_age + 2/3 * diff_workclass + 0.8 * diff_education +\
        2*diff_marital_status + diff_occupation + 1.5* diff_capital +\
        0.5*diff_native_country

def get_adult_dissimilarity_matrix(X):
    """ calculer la matrice de dissimilarité complète entre chaque individus du dataframe """
    matrix = np.zeros(shape=(len(X), len(X)), dtype=int)
    for i, j in combinations(range(len(X)), 2):
        matrix[i, j] = adult_dissimilarity(X.iloc[i], X.iloc[j])
    return matrix

#######################################################################
#                                Main                                 #
#######################################################################


def main():
    adult_clean = pd.read_csv('adult/adult_clean.csv')
    subset = adult_clean.head(n=100)
    X = subset.drop(columns=['income'])
    dissimilarity_matrix = get_adult_dissimilarity_matrix(X)
    print(dissimilarity_matrix)
    print(len(dissimilarity_matrix))


if __name__ == "__main__":
    main()




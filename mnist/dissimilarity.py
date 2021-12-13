#!/env/bin/python

import numpy as np
import pandas as pd
import textdistance

from itertools import combinations

def levenshtein(s, t):
# reprise de https://python-course.eu/applications-python/levenshtein-distance.php
# modifiée pour être applicable à des listes et non des strings
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the lists s and t.
        For all i and j, dist[i,j] will contain the Levenshtein distance between the first i characters of s and the first j characters of t
    """
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i 
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
    return dist[row][col]


def dist_mnist(img1,img2, *args, **kwargs):
    # im1 et im2 sont des matrices 14x14
    # rend la somme des distances de levenshtein des lignes ou des colonnes
    distancelignes = 0
    distancecolonnes = 0
    for i in range (0,14) :
        # somme des distances de lignes
        im1 = img1[i].tolist()
        im2 = img2[i].tolist()
        distancelignes += levenshtein(im1,im2)
        # somme des distances de colonnes -- trop complexe
        seq1 = [img1[j][i] for j in range (0,14)]
        seq2 = [img2[j][i] for j in range (0,14)]
        distancecolonnes += levenshtein(seq1, seq2)
    return min (distancecolonnes, distancelignes)

def get_matrix_from_data (dataset, line):
    # rend l'image en matrice de taille 14x14
    return dataset.iloc[line, 1:].values.reshape(14,14)

def get_mnist_dissimilarity_matrix(x): # x is the features
    # rend la matrice de distance
    distance_matrix = np.zeros((len(x), len(x)), dtype=int)
    for i, j in combinations(range(len(x)), 2):
        img1 = get_matrix_from_data(x,i)
        img2 = get_matrix_from_data(x,j)
        distance_matrix[i,j] = dist_mnist(img1,img2)
    # make matrix symetric
    i_lower = np.tril_indices(len(distance_matrix), -1)
    distance_matrix[i_lower] = distance_matrix.T[i_lower]
    # print (distance_matrix)
    return distance_matrix

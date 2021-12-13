#!/env/bin/python 

import pandas as pd 
import numpy as np

from constants import PreprocessingDict

def main():
    # set seed for reproducibility
    np.random.seed(420)

    # read raw dataset
    adult = pd.read_csv("adult/adult.csv")

    # enlever les données manquantes et les lignes duppliquées
    df_clean = adult.replace('?', np.nan).dropna().drop_duplicates()

    nb_missing_lines = len(adult) - len(adult.replace('?', np.nan).dropna())
    print(f"Le nombre de lignes avec des données manquantes est: {nb_missing_lines}")
    print(f"Le pourcentage de lignes enlevées est: {round(nb_missing_lines/len(adult) *100, 2)}%")

    # preprocessing: transformer les données
    df_clean = df_clean.replace({"workclass": PreprocessingDict.WORKCLASS_DICT})
    df_clean = df_clean.replace({"education": PreprocessingDict.EDUCATION_DICT})
    df_clean = df_clean.replace({"native-country": PreprocessingDict.COUNTRY_DICT})
    df_clean['age_label'] = df_clean['age'].apply(lambda x: get_label_age(x))
    df_clean['has_partner'] = df_clean['marital-status'].apply(
        lambda status: has_partner(status))
    df_clean = df_clean.replace({"race": PreprocessingDict.RACE_DICT})
    df_clean = df_clean.replace({"occupation": PreprocessingDict.OCCUPATION_DICT})
    df_clean['is_investing'] = df_clean.apply(
        lambda x: is_investing(x['capital-gain'], x['capital-loss']), axis=1)
    df_clean['hours_per_week_label'] = df_clean['hours-per-week'].apply(
        lambda x: get_hours_label(x))

    # drop redudant columns
    redundant_columns = ['age', 'marital-status', 'educational-num',
            'relationship', 'capital-gain', 'capital-loss', 'hours-per-week']
    df_clean.drop(redundant_columns, axis=1)

    # save clean dataset
    df_clean.to_csv('adult/adult_clean.csv', sep=',')


def get_label_age(age):
    """ donner un label à un individu selon son age """
    if age >= 16 and age <=27:
        return 'etudiant'
    elif age >= 28 and age <= 35:
        return 'premier-emploi'
    elif age >= 36 and age <= 59:
        return 'emploi-stable'
    elif age >= 60 and age <= 67:
        return 'presque-retraite'
    return 'retraite'


def get_hours_label(hours):
    """ Donner un label à un individu selon ses heures travaillées """
    if hours < 0: return "erreur"
    elif hours <= 30: return "temps-partiel"
    elif hours <= 39: return "temps-plein"
    elif hours <= 43: return "temps-plein"
    else: return "overtime"

def is_investing(gain, loss):
    """ Déterminer si un individu investi: un individu investi si son gain/perte est
    différent que 0"""
    if gain !=0 | loss != 0:
        return True
    return False

def has_partner(status):
    """ Détermine si un individu à un partenaire (retourne booléen) """
    if status in ['Married-AF-spouse', 'Married-civ-spouse']:
        return True
    return False

if __name__ == "__main__":
    main()




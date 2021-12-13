#!/env/bin/python 

""" Classes de constantes nécessaire pour calculer la dissimilarité ADULT """


#######################################################################
#                   Dictionaries for Preprocessing                    #
#######################################################################

class PreprocessingDict:
    WORKCLASS_DICT = {
        "Private": "private",
        "Self-emp-not-inc": "self-employed", 
        "Self-emp-inc": "self-employed",
        "Local-gov": "government", 
        "State-gov": "government", 
        "Federal-gov": "government", 
        "Never-worked": "jobless", 
        "Without-pay": "jobless"
    }

    EDUCATION_DICT = {
        "Preschool": "below-hs",
        "1st-4th": "below-hs", 
        "5th-6th": "below-hs",
        "7th-8th": "below-hs",
        "9th": "below-hs",
        "10th": "below-hs",
        "11th": "below-hs",
        "12th": "below-hs",
        "HS-grad": "highschool",
        "Some-college": "some-college",
        "Assoc-voc": "associate", 
        "Assoc-acdm": "associate",
        "Prof-school": "doctorate",
        "Bachelors": "bachelor",
        "Masters": "master",
        "Doctorate": "doctorate"
    }

    COUNTRY_DICT = {
        "United-States": "riche",
        "Mexico": "pauvre",
        "Philippines": "pauvre", 
        "Germany": "riche", 
        "Puerto-Rico": "pauvre", 
        "Canada": "riche", 
        "El-Salvador": "pauvre", 
        "India": "moyen", 
        "Cuba": "pauvre", 
        "England": "riche", 
        "China": "moyen", 
        "South": "moyen", 
        "Jamaica": "pauvre", 
        "Italy": "riche", 
        "Dominican-Republic": "pauvre", 
        "Japan": "riche", 
        "Guatemala": "pauvre", 
        "Poland": "moyen", 
        "Vietnam": "pauvre", 
        "Columbia": "pauvre", 
        "Haiti": "pauvre", 
        "Portugal": "riche", 
        "Taiwan": "riche",
        "Iran": "riche", 
        "Greece": "moyen", 
        "Nicaragua": "pauvre", 
        "Peru": "pauvre", 
        "Ecuador": "pauvre", 
        "France": "riche", 
        "Ireland": "riche", 
        "Hong": "pauvre", 
        "Thailand": "moyen", 
        "Cambodia": "pauvre", 
        "Trinadad&Tobago": "pauvre", 
        "Laos": "pauvre", 
        "Yugoslavia": "moyen", 
        "Outlying-US(Guam-USVI-etc)": "moyen",
        "Scotland": "riche", 
        "Honduras": "pauvre", 
        "Hungary": "moyen", 
        "Holand-Netherlands": "moyen"
    }

    RACE_DICT = {
        "Amer-Indian-Eskimo": "bipoc",
        "Asian-Pac-Islander": "non-bipoc", 
        "Black": "bipoc",
        "Other": "bipoc", 
        "White": "non-bipoc"
    }

    OCCUPATION_DICT = {
        "Priv-house-serv": "primaire", 
        "Other-service": "primaire", 
        "Handlers-cleaners": "primaire", 
        "Farming-fishing": "primaire", 
        "Machine-op-inspct": "mixte", 
        "Adm-clerical": "mixte", 
        "Transport-moving": "mixte", 
        "Craft-repair": "mixte", 
        "Sales": "tertiaire",
        "Tech-support": "tertiaire", 
        "Protective-serv": "tertiaire", 
        "Armed-Forces": "tertiaire", 
        "Prof-specialty": "tertiaire", 
        "Exec-managerial": "tertiaire"
    }


#######################################################################
#                        Dissimilarity Matrix                         #
#######################################################################

class DissimilarityMatrix:
    WORKCLASS = [
        [0,2,2,3],
        [2,0,1,3],
        [2,1,0,3], 
        [3,3,3,0] ]
    EDUCATION = [
            [0,2,2,2,4,4,5],
            [2,0,1,1,3,3,5],
            [2,1,0,1,3,3,4],
            [2,1,1,0,2,2,4],
            [4,3,3,2,0,1,2],
            [4,3,3,2,0,1,2], 
            [4,3,3,2,1,0,2], 
            [5,5,4,4,2,2,0] ]
    COUNTRY = [
            [0,1,2],
            [1,0,1],
            [2,1,0] ]
    AGE = [
            [0,2,4,3,1],
            [2,0,2,1,1],
            [4,2,0,2,3],
            [3,1,2,0,2],
            [1,1,3,2,0]
            ]
    OCCUPATION = [
            [0,1,2],
            [1,0,1],
            [2,1,0]
            ]
    HOURS = [
            [0,1,2,3],
            [1,0,1,2],
            [2,1,0,1],
            [3,2,1,0]
            ]

#######################################################################
#                     Dissimilarity Matrix Index                      #
#######################################################################

class DissimilarityMatrixIndex:
    WORKCLASS = {
            "government": 0, 
            "self-employed": 1, 
            "private": 2, 
            "jobless": 3 }
    EDUCATION = {
            "below-hs": 0, 
            "highschool": 1,
            "some-college": 2, 
            "associate": 3, 
            "bachelor": 4, 
            "master": 5, 
            "doctorate": 6
            }
    COUNTRY = {
            "riche": 0, 
            "moyen": 1, 
            "pauvre": 2
            }
    AGE = {
            "etudiant": 0,
            "premier-emploi": 1, 
            "emploi-stable": 2, 
            "presque-retraite": 3, 
            "retraite": 4
            }
    OCCUPATION = {
            "primaire": 0,
            "mixte": 1, 
            "tertiaire":2
            }
    HOURS = {
            "temps-partiel": 0,
            "temps-normal": 1, 
            "temps-plein": 2, 
            "overtime": 3
            }



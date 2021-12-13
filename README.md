# IFT 3700 - Devoir 1

## Prerequisites

```bash
# activer l'environnement virtuel
python -m venv devoir1-venv
source devoir1-venv/bin/activate

# installer les librairies
pip install --upgrade -r requirements.txt
```

## Comment exécuter le programme

```bash

# pour voir l'exploration de données (optionnel)
python3 adult/exploration.py

# générer les données propres pour MNIST et ADULT
python3 adult/preprocessing.py
python3 mnist/preprocessing.py

# exécuter les algorithmes
python3 main.py

```


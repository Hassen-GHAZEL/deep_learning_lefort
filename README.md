# TP Deep Learning

Ce projet de TP en deep learning a pour objectif d'explorer et d'évaluer différents modèles de réseaux de neurones profonds et peu profonds, en utilisant des jeux de données comme MNIST. Le projet inclut l'entraînement, l'évaluation et l'analyse des performances des modèles en fonction de divers hyperparamètres.

## Structure du projet

- `CNN.py` : Implémente un modèle de réseau de neurones convolutif (CNN) pour la classification des images MNIST.
- `Shallow_network.py` : Implémente un perceptron multicouche pour la classification des images MNIST.
- `Deep_network.py` : Contient des fonctions pour définir et entraîner des réseaux de neurones profonds.
- `Excel.py` : Gère les opérations de lecture et d'écriture dans des fichiers Excel pour enregistrer les résultats des entraînements.
- `tools.py` : Contient des fonctions utilitaires pour la gestion des hyperparamètres, la vérification de la disponibilité du GPU, l'enregistrement des logs, et l'exécution des commandes Git.
- `main1.py` à `main5.py` : Scripts principaux pour l'entraînement et l'évaluation des modèles avec différents hyperparamètres.
- `main_filter.py` : Filtre les résultats des entraînements pour extraire les meilleures performances.
- `stop_process.py` : Script pour arrêter le processus après une durée d'exécution spécifiée.
- `temperature_de_securite.py` : Vérifie la température du GPU et arrête le programme si elle dépasse un seuil de sécurité.
- `constantes.py` : Contient des constantes utilisées dans le projet.
- `cours_tuto_enonce/` : Contient des fichiers de tutoriels et d'exemples pour l'entraînement des perceptrons.
- `data/` : Contient les jeux de données utilisés pour l'entraînement des modèles.
- `excel/` : Contient les fichiers Excel pour enregistrer les résultats des entraînements.
- `image/` : Contient les images générées par les scripts (par exemple, des boxplots).
- `json/` : Contient des fichiers JSON pour enregistrer des informations comme le PID du programme.
- `txt/` : Contient des fichiers texte pour enregistrer les logs du programme.

## Réseaux de Neurones Convolutifs (CNN)

Les réseaux de neurones convolutifs (CNN) sont utilisés pour la classification des images dans ce projet. Le fichier [CNN.py](CNN.py) implémente un modèle CNN pour la classification des images MNIST. Le modèle est construit avec plusieurs couches de convolution, de pooling et de couches entièrement connectées. Les principales étapes incluent :

1. **Construction du modèle** : La méthode `build_model` définit l'architecture du CNN avec des couches de convolution, de normalisation par lots, de pooling, et des couches entièrement connectées.
2. **Entraînement et évaluation** : La méthode `train_and_evaluate` entraîne le modèle sur les données d'entraînement et l'évalue sur les ensembles de validation et de test. Les résultats sont enregistrés dans un fichier Excel.
3. **Évaluation** : La méthode `evaluate` évalue le modèle sur un ensemble de données donné et retourne la perte et l'accuracy.

## Installation

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```sh
pip install -r requirements.txt
```
import gzip
import torch
from torch.utils.data import TensorDataset
from CNN import MNISTModel
from tools import charger_donnees, definir_hyperparametres
from Excel import ExcelManager
from datetime import datetime
from tools import calculer_ecart_temps, enregistrer_debut_programme, enregistrer_fin_programme, git_commit_and_push, shutdown_system
from constantes import *

def evaluer_hyperparametre(nom, valeurs, params):
    global excel  # Accès à la variable globale excel pour l'enregistrement des résultats
    print(f"Influence de {nom} sur le modèle :")  # Affiche l'hyperparamètre en cours d'évaluation
    for valeur in valeurs:  # Boucle à travers les différentes valeurs de l'hyperparamètre
        heure_debut_iteration = datetime.now().strftime("%H:%M:%S")  # Enregistre l'heure de début de l'itération
        print(f"\t{nom} : {valeur}")  # Affiche la valeur de l'hyperparamètre testé

        # Met à jour les paramètres avec la valeur actuelle
        params[nom.lower()] = valeur
        print(f"\tHyperparamètres : {params['batch_size']}, {params['learning_rate']}")  # Affiche les paramètres mis à jour

        # Charge les jeux de données d'entraînement et de test avec validation
        train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)

        # Initialise le modèle avec les paramètres définis
        model = MNISTModel(
            num_classes=params["output_size"],
            batch_size=params['batch_size'],  # Taille du lot pour l'entraînement
            learning_rate=params['learning_rate'],  # Taux d'apprentissage pour l'optimiseur
            nb_epochs=params['nb_epochs'],  # Nombre d'époques pour l'entraînement
            excel=excel  # Instance de ExcelManager pour l'enregistrement des résultats
        )

        # Entraîne et évalue le modèle sur les jeux de données chargés
        model.train_and_evaluate(nom, train_loader, val_loader, test_loader, is_nested=False)

        heure_fin_iteration = datetime.now().strftime("%H:%M:%S")  # Enregistre l'heure de fin de l'itération
        ecart = calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)  # Calcule le temps écoulé
        print(f"\tDurée de cette itération ({nom}={valeur}): {ecart}")  # Affiche la durée de l'itération

if __name__ == '__main__':
    # Obtenir l'heure de début du programme
    enregistrer_debut_programme()

    # Chargement des données à partir d'un fichier gzip
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Préparation des jeux de données en créant des ensembles de données à partir des tensors
    train_dataset = TensorDataset(data_train, label_train)  # Ensemble d'entraînement
    test_dataset = TensorDataset(data_test, label_test)  # Ensemble de test

    # Définition des paramètres initiaux pour le modèle
    params = definir_hyperparametres()

    # Noms des colonnes pour le fichier Excel
    column_names = ["numéro epoch", "nb_epochs"] + ["batch_size", "learning_rate"] + ["Training Loss", "Validation Loss", "Test Loss", "Accuracy"]

    # Initialisation de la gestion du fichier Excel
    excel = ExcelManager("excel/CNN.xlsx", column_names)

    # Évaluation des hyperparamètres avec les valeurs définies
    evaluer_hyperparametre("BATCH_SIZE", tab_batch_size, params)  # Évalue l'hyperparamètre batch_size
    evaluer_hyperparametre("LEARNING_RATE", tab_learning_rate, params)  # Évalue l'hyperparamètre learning_rate

    # Enregistrement du temps total d'exécution et des résultats dans Excel
    enregistrer_fin_programme()
    git_commit_and_push("Analyse des hyperparamètres pour le modèle CNN terminée")  # Commit des résultats dans Git
    # shutdown_system()  # Décommentez pour éteindre le système après l'exécution

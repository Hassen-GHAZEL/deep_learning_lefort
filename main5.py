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
    global excel
    print(f"Influence de {nom} sur le modèle :")
    for valeur in valeurs:
        heure_debut_iteration = datetime.now().strftime("%H:%M:%S")
        print(f"\t{nom} : {valeur}")

        # Mettre à jour les paramètres avec la valeur testée
        params[nom.lower()] = valeur
        print(f"\tHyperparamètres : {params['batch_size']}, {params['learning_rate']}")

        # Charger les données d'entraînement et de test avec validation
        train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)

        # Initialiser le modèle
        model = MNISTModel(
            num_classes=params["output_size"],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            nb_epochs=params['nb_epochs'],
            excel=excel
        )

        # Entraîner et évaluer le modèle
        model.train_and_evaluate(nom, train_loader, val_loader, test_loader, is_nested=False)

        heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
        ecart = calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)
        print(f"\tDurée de cette itération ({nom}={valeur}): {ecart}")

if __name__ == '__main__':
    # Obtenir l'heure de début
    enregistrer_debut_programme()

    # Chargement des données
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Préparation des jeux de données
    train_dataset = TensorDataset(data_train, label_train)  # Jeu d'entraînement
    test_dataset = TensorDataset(data_test, label_test)  # Jeu de test

    # Définition des paramètres initiaux
    params = definir_hyperparametres()

    # Noms des colonnes pour Excel
    column_names = ["numéro epoch", "nb_epochs"] + ["batch_size", "learning_rate"] + ["Training Loss", "Validation Loss",
                                                                                "Test Loss", "Accuracy"]

    # Initialisation de la gestion du fichier Excel
    excel = ExcelManager("excel/CNN.xlsx", column_names)

    # Définition des valeurs à tester pour chaque hyperparamètre
    # Exemple d'évaluation des hyperparamètres
    evaluer_hyperparametre("BATCH_SIZE", tab_batch_size, params)
    evaluer_hyperparametre("LEARNING_RATE", tab_learning_rate, params)

    # Calculer et afficher le temps total d'exécution
    enregistrer_fin_programme()
    git_commit_and_push("Analyse des hyperparamètres pour le modèle CNN terminée")
    # shutdown_system()

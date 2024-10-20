import gzip
from Shallow_network import *
from torch.utils.data import TensorDataset
from Excel import ExcelManager
from tools import *
from constantes import *
from datetime import datetime


def evaluer_hyperparametre(nom, valeurs):
    global excel  # Accès à la variable ExcelManager globale
    print(f"Influence de {nom} sur le modèle :")  # Affichage du paramètre à évaluer
    for valeur in valeurs:  # Boucle sur les différentes valeurs
        heure_debut_iteration = datetime.now().strftime("%H:%M:%S")  # Horodatage de début
        print(f"\t{nom} : {valeur}")  # Affichage de la valeur actuelle

        # Définition des hyperparamètres
        params = definir_hyperparametres(**{nom.lower(): valeur})
        print(f"\tHyperparamètres : {params}")  # Affichage des hyperparamètres

        # Initialisation du modèle
        model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'],
                                      params['weight_init_range'], excel)

        # Chargement des données pour l'entraînement, la validation et le test
        train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)
        # Entraînement et évaluation du modèle
        model.train_and_evaluate(nom, train_loader, val_loader, test_loader, params, is_nested=False)

        heure_fin_iteration = datetime.now().strftime("%H:%M:%S")  # Horodatage de fin
        ecart = calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)  # Calcul du temps écoulé
        print(f"\tDurée de cette itération ({nom}={valeur}): {ecart}")  # Affichage de la durée


if __name__ == '__main__':
    enregistrer_debut_programme()  # Enregistrement du début du programme

    # Chargement des données
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Création des jeux de données
    train_dataset = TensorDataset(data_train, label_train)  # Jeu d'entraînement
    test_dataset = TensorDataset(data_test, label_test)  # Jeu de test

    # Noms des colonnes pour le fichier Excel
    column_names = ["numéro epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss",
                                                                                "Test Loss", "Accuracy"]

    # Initialisation du gestionnaire de fichier Excel
    excel = ExcelManager("excel/test2.xlsx", column_names)

    # Évaluation des hyperparamètres
    evaluer_hyperparametre("BATCH_SIZE", tab_batch_size)
    evaluer_hyperparametre("LEARNING_RATE", tab_learning_rate)
    evaluer_hyperparametre("HIDDEN_SIZE", tab_hidden_size)
    evaluer_hyperparametre("WEIGHT_INIT_RANGE", tab_weight_init_range)

    enregistrer_fin_programme()  # Enregistrement de la fin du programme
    shutdown_system()  # éteindre le système

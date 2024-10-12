import gzip  # Assurez-vous d'importer le module gzip
from Deep_network import * # Assurez-vous que cette classe est correctement définie
from torch.utils.data import TensorDataset, random_split
from Excel import ExcelManager
from tools import *
from constantes import *
from datetime import datetime


def evaluer_hyperparametre(nom, valeurs):
    global excel
    print(f"Influence de {nom} sur le modèle :")
    for valeur in valeurs:
        heure_debut_iteration = datetime.now().strftime("%H:%M:%S")
        print(f"\t{nom} : {valeur}")

        params = definir_hyperparametres(**{nom.lower(): valeur})
        print(f"\tHyperparamètres : {params}")

        use_gpu = get_gpu_temperature() < 50
        model = DeepNetwork(params['input_size'], params['hidden_size'], params['output_size'],
                                      params['weight_init_range'], excel, use_gpu)

        if params['hidden_size'] == 2:
            params['nb_epochs'] = 22
        elif params['hidden_size'] == 3:
            params['nb_epochs'] = 35

        # Inclure val_loader pour le jeu de validation
        train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)
        model.train_and_evaluate(nom, train_loader, val_loader, test_loader, params, is_nested=False)

        heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
        ecart = calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)
        print(f"\tDurée de cette itération ({nom}={valeur}): {ecart}")



if __name__ == '__main__':
    # # Obtenir l'heure de début
    enregistrer_debut_programme()
    #
    # Chargement des données
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Préparation des jeux de données
    train_dataset = TensorDataset(data_train, label_train)  # Jeu d'entraînement
    test_dataset = TensorDataset(data_test, label_test)  # Jeu de test

    # Noms des colonnes pour Excel
    column_names = ["numéro epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss",
                                                                                "Test Loss", "Accuracy"]

    # Initialisation de la gestion du fichier Excel
    excel = ExcelManager("excel/deep_network.xlsx", column_names)


    # Définition des valeurs à tester pour chaque hyperparamètre
    # evaluer_hyperparametre("BATCH_SIZE", tab_batch_size)
    # evaluer_hyperparametre("LEARNING_RATE", tab_learning_rate)
    evaluer_hyperparametre("HIDDEN_SIZE", tab_hidden_size)
    # evaluer_hyperparametre("WEIGHT_INIT_RANGE", tab_weight_init_range)


    # Calculer et afficher le temps total d'exécution
    enregistrer_fin_programme()
    git_commit_and_push("deep network, analyse hyperparamètres terminée")
    # shutdown_system()

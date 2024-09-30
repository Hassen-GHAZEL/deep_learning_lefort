import gzip  # Assurez-vous d'importer le module gzip
from Shallow_network import *  # Assurez-vous que cette classe est correctement définie
from torch.utils.data import TensorDataset
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
        model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'],
                                      params['weight_init_range'], excel)

        # Remplacez val_loader par test_loader pour utiliser le bon jeu de test
        train_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)
        model.train_and_evaluate(nom, train_loader, test_loader, params, is_nested=False)

        heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
        ecart = calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)
        print(f"\tDurée de cette itération ({nom}={valeur}): {ecart}")


if __name__ == '__main__':
    # Obtenir l'heure de début
    enregistrer_debut_programme()

    # Chargement des données
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Préparation des jeux de données
    train_dataset = TensorDataset(data_train, label_train)  # Jeu d'entraînement
    test_dataset = TensorDataset(data_test, label_test)  # Jeu de test (correctement nommé)

    column_names = ["numéro epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Test Loss",
                                                                                "Accuracy"]

    excel = ExcelManager("tableau.xlsx", column_names)

    # Définition des valeurs à tester pour chaque hyperparamètre
    evaluer_hyperparametre("BATCH_SIZE", tab_batch_size)
    evaluer_hyperparametre("NB_EPOCHS", tab_nb_epochs)
    evaluer_hyperparametre("LEARNING_RATE", tab_learning_rate)
    evaluer_hyperparametre("HIDDEN_SIZE", tab_hidden_size)
    evaluer_hyperparametre("WEIGHT_INIT_RANGE", tab_weight_init_range)

    # Obtenir l'heure de fin
    heure_de_fin = datetime.now().strftime("%H:%M:%S")

    # Calculer et afficher le temps total d'exécution
    enregistrer_fin_programme()
    shutdown_system()

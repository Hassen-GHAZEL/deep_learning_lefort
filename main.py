import gzip  # Assurez-vous d'importer le module gzip
from Shallow_network import *
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
        train_loader, val_loader = charger_donnees(train_dataset, val_dataset, params)
        model.train_and_evaluate(nom, train_loader, val_loader, params, is_nested=False)

        heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
        ecart = calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)
        print(f"\tDurée de cette itération ({nom}={valeur}): {ecart}")

if __name__ == '__main__':

    # Obtenir l'heure de début
    heure_de_debut = datetime.now().strftime("%H:%M:%S")

    # Chargement des données
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Préparation des jeux de données
    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(data_test, label_test)

    column_names = ["numéro epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss", "Accuracy"]

    excel = ExcelManager("tableau.xlsx", column_names)

    # Définition des valeurs à tester pour chaque hyperparamètre
    evaluer_hyperparametre("BATCH_SIZE", tab_batch_size)
    evaluer_hyperparametre("NB_EPOCHS", tab_nb_epochs)
    #evaluer_hyperparametre("LEARNING_RATE", tab_learning_rate)
    #evaluer_hyperparametre("HIDDEN_SIZE", tab_hidden_size)
    #evaluer_hyperparametre("WEIGHT_INIT_RANGE", tab_weight_init_range)

    # Obtenir l'heure de fin
    heure_de_fin = datetime.now().strftime("%H:%M:%S")

    # Calculer et afficher le temps total d'exécution
    duree_totale = calculer_ecart_temps(heure_de_debut, heure_de_fin)
    msg = f"heure de début : {heure_de_debut}, heure de fin : {heure_de_fin}, durée totale : {duree_totale} + 01:19:30 = ?"
    create_or_overwrite_file("duree_totale.txt", msg)
    shutdown_system()
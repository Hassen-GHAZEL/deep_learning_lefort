import gzip
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from datetime import datetime
from Shallow_network import PerceptronMulticouche
from Excel import ExcelManager
from tools import *
from constantes import *

def load_data():
    """Charge et retourne les jeux de données d'entraînement, de validation et de test."""
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Séparation du jeu d'entraînement en sous-ensemble d'entraînement et de validation
    train_size = int(0.8 * len(data_train))  # 80% pour l'entraînement
    val_size = len(data_train) - train_size  # 20% pour la validation
    train_dataset, val_dataset = random_split(TensorDataset(data_train, label_train), [train_size, val_size])

    test_dataset = TensorDataset(data_test, label_test)  # Jeu de test

    return train_dataset, val_dataset, test_dataset

def main():
    # Obtenir l'heure de début
    heure_de_debut = datetime.now().strftime("%H:%M:%S")

    nb_operation = len(tab_batch_size) * len(tab_learning_rate) * len(tab_hidden_size) * len(tab_weight_init_range)
    print(f"{len(tab_batch_size)} * {len(tab_learning_rate)} * {len(tab_hidden_size)} * {len(tab_weight_init_range)} = {nb_operation}")

    # Chargement des données
    train_dataset, val_dataset, test_dataset = load_data()

    # Assertions pour vérifier que les jeux de données sont correctement nommés
    assert train_dataset is not None, "Le jeu de données d'entraînement ne doit pas être nul."
    assert val_dataset is not None, "Le jeu de validation ne doit pas être nul."
    assert test_dataset is not None, "Le jeu de données de test ne doit pas être nul."

    # Optionnel : Vérifier les dimensions des jeux de données
    assert len(train_dataset) > 0, "Le jeu de données d'entraînement doit contenir des exemples."
    assert len(val_dataset) > 0, "Le jeu de validation doit contenir des exemples."
    assert len(test_dataset) > 0, "Le jeu de données de test doit contenir des exemples."

    column_names = list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss", "Test Loss", "Accuracy"]
    excel = ExcelManager("tableau_combinaison.xlsx", column_names)

    nb_row_in_excel = excel.count_rows("EVERYTHING")
    default_params = definir_hyperparametres()

    i = 1
    bool = True
    compt_repetitions = 0

    for weight_init_range in tab_weight_init_range:
        for hidden_size in tab_hidden_size:
            for learning_rate in tab_learning_rate:
                for batch_size in tab_batch_size:
                    if i < nb_row_in_excel:
                        print(f"Iteration {i} déjà faite")
                        i += 1
                        continue

                    # Vérification si les hyperparamètres sont égaux aux valeurs par défaut
                    is_default = (batch_size == default_params['batch_size'] and
                                  learning_rate == default_params['learning_rate'] and
                                  hidden_size == default_params['hidden_size'] and
                                  weight_init_range == default_params['weight_init_range'])

                    if is_default and not bool:
                        i += 1
                        continue

                    heure_debut_iteration = datetime.now().strftime("%H:%M:%S")
                    print(f"Iteration {i}/{nb_operation} ({i * 100 / nb_operation:.3f}%), heure de début itération : {heure_debut_iteration}")
                    print(f"BATCH_SIZE: {batch_size}, LEARNING_RATE: {learning_rate}, HIDDEN_SIZE: {hidden_size}, WEIGHT_INIT_RANGE: {weight_init_range}")

                    # Chargement des hyperparamètres
                    params = definir_hyperparametres(
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                        weight_init_range=weight_init_range
                    )

                    # Initialisation du modèle
                    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'],
                                                  params['weight_init_range'], excel)

                    # Chargement des données (ajout du jeu de validation)
                    train_loader, val_loader, test_loader = charger_donnees(train_dataset, val_dataset, test_dataset, params)

                    # Entraînement et évaluation du modèle
                    model.train_and_evaluate("EVERYTHING", train_loader, val_loader, test_loader, params)

                    heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
                    print(f"Heure de fin itération : {heure_fin_iteration}")
                    print(f"Durée de l'itération : {calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)}")

                    if is_default:
                        bool = False
                        compt_repetitions += 1

                    # Incrémenter le compteur
                    i += 1

    heure_de_fin = datetime.now().strftime("%H:%M:%S")
    msg = f"576 - 600 heure de début : {heure_de_debut}, heure de fin : {heure_de_fin}, durée totale : {calculer_ecart_temps(heure_de_debut, heure_de_fin)}"
    create_or_overwrite_file("duree_totale_combinaison.txt", msg)

if __name__ == '__main__':
    main()

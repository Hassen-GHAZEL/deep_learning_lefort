import gzip
from torch.utils.data import TensorDataset
from Shallow_network import PerceptronMulticouche
from Excel import ExcelManager
from tools import *
from constantes import *


def charger_datasets(filepath):
    """Charge les ensembles d'entraînement et de test à partir d'un fichier compressé."""
    with gzip.open(filepath, 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)
    return TensorDataset(data_train, label_train), TensorDataset(data_test, label_test)


if __name__ == '__main__':

    enregistrer_debut_programme()

    # Initialisation Excel
    column_names = ["numeor epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss", "Test Loss", "Accuracy"]
    excel = ExcelManager("excel/shallow_network_combinaison.xlsx", column_names)

    # Chargement des données une seule fois
    filepath = 'data/mnist.pkl.gz'
    train_dataset, test_dataset = charger_datasets(filepath)

    # Chargement des hyperparamètres par défaut
    default_params = definir_hyperparametres()

    nb_row_in_excel = excel.count_rows("EVERYTHING")
    if nb_row_in_excel > 0 and ((nb_row_in_excel - 1) % 10) != 0:
        raise Exception("Une epoch ou plusieurs n'ont pas été enregistrées")

    nb_combinaison = len(tab_batch_size) * len(tab_learning_rate) * len(tab_hidden_size) * len(tab_weight_init_range)
    print(f"{len(tab_batch_size)} * {len(tab_learning_rate)} * {len(tab_hidden_size)} * {len(tab_weight_init_range)} = {nb_combinaison}")

    i = 1
    bool_default = True
    compt_repetitions = 0

    # Boucle pour tester les combinaisons d'hyperparamètres
    for weight_init_range in tab_weight_init_range:
        for hidden_size in tab_hidden_size:
            for learning_rate in tab_learning_rate:
                for batch_size in tab_batch_size:

                    if i <= (nb_row_in_excel - 1) / default_params['nb_epochs']:
                        print(f"Iteration {i} déjà faite")
                        i += 1
                        continue

                    # Chargement des hyperparamètres pour cette itération
                    params = definir_hyperparametres(
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                        weight_init_range=weight_init_range
                    )

                    # Création des DataLoaders (jeu d'entraînement et jeu de test sont constants)
                    train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)

                    # Initialisation du modèle avec les hyperparamètres actuels
                    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'],
                                                  params['weight_init_range'], excel)

                    # Entraînement et évaluation du modèle
                    model.train_and_evaluate("EVERYTHING", train_loader, val_loader, test_loader, params)

                    if (batch_size == default_params['batch_size'] and
                        learning_rate == default_params['learning_rate'] and
                        hidden_size == default_params['hidden_size'] and
                        weight_init_range == default_params['weight_init_range']) and not bool_default:
                        bool_default = False
                        compt_repetitions += 1

                    # Incrémenter le compteur
                    i += 1

    enregistrer_fin_programme()
    git_commit_and_push("Toutes les combinaisons d'hyperparamètres ont été testées !")
    shutdown_system()

import gzip
from torch.utils.data import TensorDataset
from Deep_network import DeepNetwork
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
    column_names = ["numero epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss", "Test Loss", "Accuracy"]
    excel = ExcelManager("excel/deep_network_combinaison2.xlsx", column_names)

    # Chargement des données une seule fois
    filepath = 'data/mnist.pkl.gz'
    train_dataset, test_dataset = charger_datasets(filepath)

    # Chargement des hyperparamètres par défaut
    default_params = definir_hyperparametres()

    nb_row_in_excel = excel.count_rows("EVERYTHING")
    if nb_row_in_excel > 0 and ((nb_row_in_excel - 1) % 10) != 0:
        raise Exception("Une epoch ou plusieurs n'ont pas été enregistrées pour le dernier modèle !")

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

                    # Vérification si les hyperparamètres sont égaux aux valeurs par défaut
                    is_default = (batch_size == default_params['batch_size'] and
                                  learning_rate == default_params['learning_rate'] and
                                  hidden_size == default_params['hidden_size'] and
                                  weight_init_range == default_params['weight_init_range'])

                    if is_default and not bool_default:
                        i += 1
                        continue

                    heure_debut_iteration = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"ITERATION {i}/{nb_combinaison} ({i * 100 / nb_combinaison:.3f}%), heure de début itération : {heure_debut_iteration}")
                    print(
                        f"batch_size : {batch_size}, learning_rate : {learning_rate}, hidden_size : {hidden_size}, weight_init_range : {weight_init_range}")

                    # Chargement des hyperparamètres pour cette itération
                    params = definir_hyperparametres(
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                        weight_init_range=weight_init_range
                    )

                    use_GPU = get_gpu_temperature() < 50


                    # Initialisation du modèle avec les hyperparamètres actuels
                    model = DeepNetwork(params['input_size'], params['hidden_size'], params['output_size'],
                                                  params['weight_init_range'], excel, True)

                    # Création des DataLoaders (jeu d'entraînement et jeu de test sont constants)
                    train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)

                    # Entraînement et évaluation du modèle
                    model.train_and_evaluate("EVERYTHING", train_loader, val_loader, test_loader, params, False)

                    heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
                    print(f"Heure de fin itération : {heure_fin_iteration}")
                    print(f"Durée de l'itération : {calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)} \n")

                    if is_default:
                        bool = False
                        compt_repetitions += 1

                    # Incrémenter le compteur
                    i += 1

    create_or_overwrite_file("txt/info.txt", f"Nombre de répétitions des hyperparamètres par défaut : {compt_repetitions}")
    enregistrer_fin_programme()
    git_commit_and_push("Toutes les combinaisons d'hyperparamètres ont été testées pour deep_network, y compris (-0.1, 0.1) !")
    # shutdown_system()

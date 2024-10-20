import gzip
from torch.utils.data import TensorDataset
from Shallow_network import PerceptronMulticouche
from Excel import ExcelManager
from tools import *
from constantes import *


def charger_datasets(filepath):
    """Charge les ensembles d'entraînement et de test à partir d'un fichier compressé."""
    with gzip.open(filepath, 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)  # Chargement des données
    return TensorDataset(data_train, label_train), TensorDataset(data_test, label_test)  # Retourne les jeux de données


if __name__ == '__main__':

    enregistrer_debut_programme()  # Enregistrement du début du programme

    # Initialisation Excel
    column_names = ["numeor epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss", "Test Loss", "Accuracy"]
    excel = ExcelManager("excel/shallow_network_combinaison.xlsx", column_names)  # Création du gestionnaire Excel

    # Chargement des données une seule fois
    filepath = 'data/mnist.pkl.gz'
    train_dataset, test_dataset = charger_datasets(filepath)  # Chargement des ensembles de données

    # Chargement des hyperparamètres par défaut
    default_params = definir_hyperparametres()

    # Vérification de l'enregistrement des epochs dans le fichier Excel
    nb_row_in_excel = excel.count_rows("EVERYTHING")
    if nb_row_in_excel > 0 and ((nb_row_in_excel - 1) % 10) != 0:
        raise Exception("Une epoch ou plusieurs n'ont pas été enregistrées pour le dernier modèle !")

    # Calcul du nombre total de combinaisons d'hyperparamètres
    nb_combinaison = len(tab_batch_size) * len(tab_learning_rate) * len(tab_hidden_size) * len(tab_weight_init_range)
    print(f"{len(tab_batch_size)} * {len(tab_learning_rate)} * {len(tab_hidden_size)} * {len(tab_weight_init_range)} = {nb_combinaison}")

    i = 1
    bool_default = True  # Indicateur pour les hyperparamètres par défaut
    compt_repetitions = 0  # Compteur pour les répétitions des hyperparamètres par défaut

    # Boucle pour tester les combinaisons d'hyperparamètres
    for weight_init_range in tab_weight_init_range:
        for hidden_size in tab_hidden_size:
            for learning_rate in tab_learning_rate:
                for batch_size in tab_batch_size:

                    # Vérification si l'itération a déjà été effectuée
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

                    heure_debut_iteration = datetime.now().strftime("%H:%M:%S")  # Horodatage de début de l'itération
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

                    # Initialisation du modèle avec les hyperparamètres actuels
                    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'],
                                                  params['weight_init_range'], excel)

                    # Création des DataLoaders (jeu d'entraînement et jeu de test sont constants)
                    train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)

                    # Entraînement et évaluation du modèle
                    model.train_and_evaluate("EVERYTHING", train_loader, val_loader, test_loader, params)

                    heure_fin_iteration = datetime.now().strftime("%H:%M:%S")  # Horodatage de fin de l'itération
                    print(f"Heure de fin itération : {heure_fin_iteration}")
                    print(f"Durée de l'itération : {calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)} \n")

                    if is_default:
                        bool_default = False  # Modification de l'indicateur si les hyperparamètres sont par défaut
                        compt_repetitions += 1  # Incrémenter le compteur

                    # Incrémenter le compteur
                    i += 1

    create_or_overwrite_file("txt/info.txt", f"Nombre de répétitions des hyperparamètres par défaut : {compt_repetitions}")
    enregistrer_fin_programme()  # Enregistrement de la fin du programme
    git_commit_and_push("Toutes les combinaisons d'hyperparamètres ont été testées !")  # Commit des résultats
    shutdown_system()  # Arrêt du système

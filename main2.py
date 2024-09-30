import gzip
from torch.utils.data import TensorDataset
from Shallow_network import PerceptronMulticouche
from Excel import ExcelManager
from tools import *
from constantes import *

def load_data():
    """Charge et retourne les jeux de données d'entraînement et de test."""
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)
    return TensorDataset(data_train, label_train), TensorDataset(data_test, label_test)

def main():

    enregistrer_debut_programme()

    nb_operation = len(tab_batch_size) * len(tab_learning_rate) * len(tab_hidden_size) * len(tab_weight_init_range)
    print(f"{len(tab_batch_size)} * {len(tab_learning_rate)} * {len(tab_hidden_size)} * {len(tab_weight_init_range)} = {nb_operation}")

    # Chargement des données
    train_dataset, test_dataset = load_data()

    # Assertions pour vérifier que les jeux de données sont correctement nommés
    assert train_dataset is not None, "Le jeu de données d'entraînement ne doit pas être nul."
    assert test_dataset is not None, "Le jeu de données de test ne doit pas être nul."

    # Optionnel : Vérifier les dimensions des jeux de données
    assert len(train_dataset.tensors[0]) > 0, "Le jeu de données d'entraînement doit contenir des exemples."
    assert len(test_dataset.tensors[0]) > 0, "Le jeu de données de test doit contenir des exemples."

    column_names = list(definir_hyperparametres().keys()) + ["Training Loss", "Test Loss", "Accuracy"]
    excel = ExcelManager("shallow_network_imbrique.xlsx", column_names)

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
                    print(f"Iteration {i}/{nb_operation}({i * 100 / nb_operation:.3f}%), heure de début itération : {heure_debut_iteration}")
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

                    # Chargement des données
                    train_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)

                    # Entraînement du modèle
                    model.train_and_evaluate("EVERYTHING", train_loader, test_loader, params, is_nested=False)

                    heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
                    print(f"heure de fin iteration : {heure_fin_iteration}")
                    print(f"duree iteration : {calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)}")

                    if is_default:
                        bool = False
                        compt_repetitions += 1

                    # Incrémenter le compteur
                    i += 1

    enregistrer_fin_programme()

if __name__ == '__main__':
    main()

import gzip  # Assurez-vous d'importer le module gzip
from Shallow_network import *
from torch.utils.data import TensorDataset
from Excel import ExcelManager
from tools import *
from constantes import *
from datetime import datetime



if __name__ == '__main__':

    # Obtenir l'heure de début
    heure_de_debut = datetime.now().strftime("%H:%M:%S")

    nb_operation = len(tab_batch_size) * len(tab_learning_rate) * len(tab_hidden_size) * len(tab_weight_init_range)
    print(f"{len(tab_batch_size)} * {len(tab_learning_rate)} * {len(tab_hidden_size)} * {len(tab_weight_init_range)} = {nb_operation}")

    # Chargement des données
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Préparation des jeux de données
    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(data_test, label_test)

    column_names = list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss", "Accuracy"]
    # Boucles imbriquées pour tester toutes les combinaisons d'hyperparamètres

    excel = ExcelManager("tableau_combinaison.xlsx", column_names)

    nb_row_in_excel = excel.count_rows("EVERYTHING")

    default_params = definir_hyperparametres()

    compt_repetitions = 0

    i=1

    bool = True

    for weight_init_range in tab_weight_init_range:
        for hidden_size in tab_hidden_size:
            for learning_rate in tab_learning_rate:
                for batch_size in tab_batch_size:
                    if i < nb_row_in_excel:
                        print(f"iteration {i} deja faite")
                        i+=1
                        continue
                    # Vérification si les hyperparamètres sont égaux aux valeurs par défaut
                    is_default = (batch_size == default_params['batch_size'] and
                                  learning_rate == default_params['learning_rate'] and
                                  hidden_size == default_params['hidden_size'] and
                                  weight_init_range == default_params['weight_init_range'])

                    if is_default and not bool:
                        compt_repetitions += 1
                        continue


                    heure_debut_iteration = datetime.now().strftime("%H:%M:%S")
                    print(f"Iteration {i}/{nb_operation}({i*100/nb_operation:.3f}%), heure de debut iteration : {heure_debut_iteration}")
                    print(
                        f"BATCH_SIZE: {batch_size}, LEARNING_RATE: {learning_rate}, HIDDEN_SIZE: {hidden_size}, WEIGHT_INIT_RANGE: {weight_init_range}")

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
                    train_loader, val_loader = charger_donnees(train_dataset, val_dataset, params)

                    # Entraînement du modèle
                    model.train_and_evaluate("EVERYTHING", train_loader, val_loader, params)

                    heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
                    print(f"heure de fin iteration : {heure_fin_iteration}")
                    print(f"duree iteration : {calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)}")

                    if is_default:
                        bool = False
                        compt_repetitions += 1

                    # Incrémenter le compteur
                    i += 1
    heure_de_fin = datetime.now().strftime("%H:%M:%S")
    msg = f"576 - 600 heure de debut : {heure_de_debut}, heure de fin : {heure_de_fin}, duree totale : {calculer_ecart_temps(heure_de_debut, heure_de_fin)}"
    create_or_overwrite_file("duree_totale_combinaison.txt", msg)
    #shutdown_system()
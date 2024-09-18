from datetime import datetime, timedelta
from Shallow_network import *


#virer l'insertion des ligne avant echo final et echo_iter = 0

# Obtenir l'heure de début
heure_de_debut = datetime.now().strftime("%H:%M:%S")

# Définition des valeurs à tester pour chaque hyperparamètre
tab_batch_size = list(range(1, 21, 2))
tab_nb_epochs = list(range(1, 21, 2))
tab_learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
tab_hidden_size = [32, 64, 128, 256, 512]
tab_weight_init_range = [(0, 0.1), (-0.1, 0.1), (-0.01, 0.01), (-0.001, 0.001), (-0.0001, 0.0001)]


nb_operation = len(tab_batch_size) * len(tab_learning_rate) * len(tab_hidden_size) * len(tab_weight_init_range) * len(tab_nb_epochs)

# Chargement des données
with gzip.open('mnist.pkl.gz', 'rb') as f:
    (data_train, label_train), (data_test, label_test) = torch.load(f)

# Préparation des jeux de données
train_dataset = TensorDataset(data_train, label_train)
val_dataset = TensorDataset(data_test, label_test)  # Utilisation de l'ensemble de test comme validation pour simplifier

def charger_donnees(params):
    """
    Fonction pour charger les données et créer les DataLoader.
    """
    # Création des DataLoader
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    return train_loader, val_loader


# Initialiser le compteur d'itérations
i = 1
bool = True
compt_repetitions = 0
default_params = definir_hyperparametres()
print(f"PerceptronMulticouche.last_row : {PerceptronMulticouche.last_row}")
# Boucles imbriquées pour tester toutes les combinaisons d'hyperparamètres
for weight_init_range in tab_weight_init_range:
    for hidden_size in tab_hidden_size:
        for learning_rate in tab_learning_rate:
            for nb_epochs in tab_nb_epochs:
                for batch_size in tab_batch_size:
                    if i<=PerceptronMulticouche.last_row:
                        print(f"iteration {i} deja faite")
                        i+=1
                        continue
                    # Vérification si les hyperparamètres sont égaux aux valeurs par défaut
                    is_default = (batch_size == default_params['batch_size'] and
                                  nb_epochs == default_params['nb_epochs'] and
                                  learning_rate == default_params['learning_rate'] and
                                  hidden_size == default_params['hidden_size'] and
                                  weight_init_range == default_params['weight_init_range'])

                    if is_default and not bool:
                        continue


                    print(f"Iteration {i}/{nb_operation}({i*100/nb_operation:.3f}%):")
                    print(
                        f"BATCH_SIZE: {batch_size}, NB_EPOCHS: {nb_epochs}, LEARNING_RATE: {learning_rate}, HIDDEN_SIZE: {hidden_size}, WEIGHT_INIT_RANGE: {weight_init_range}")

                    # Chargement des hyperparamètres
                    params = definir_hyperparametres(
                        batch_size=batch_size,
                        nb_epochs=nb_epochs,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                        weight_init_range=weight_init_range
                    )

                    # Initialisation du modèle
                    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'],
                                                  params['weight_init_range'])

                    # Chargement des données
                    train_loader, val_loader = charger_donnees(params)

                    # Entraînement du modèle
                    model.train_and_evaluate(train_loader, val_loader, i, params, nb_operation, "EVERYTHING")

                    if is_default:
                        bool = False
                        compt_repetitions += 1

                    # Incrémenter le compteur
                    i += 1



print(f"Nombre de répétitions de l'itération par défaut : {compt_repetitions}")
# Obtenir l'heure de fin
heure_de_fin = datetime.now().strftime("%H:%M:%S")

# Convertir les chaînes en objets datetime pour pouvoir calculer l'écart
format_heure = "%H:%M:%S"
t1 = datetime.strptime(heure_de_debut, format_heure)
t2 = datetime.strptime(heure_de_fin, format_heure)

# Si l'heure de fin est plus petite que l'heure de début, ajouter un jour entier à t2
if t2 < t1:
    t2 += timedelta(days=1)

# Calculer l'écart
ecart = t2 - t1

# Extraire les heures, minutes et secondes de l'écart
heures, reste = divmod(ecart.seconds, 3600)
minutes, secondes = divmod(reste, 60)

# Afficher l'écart sous forme "%H:%M:%S"
print(f"heure de début : {heure_de_debut}, heure de fin : {heure_de_fin}, durée : {heures:02d}:{minutes:02d}:{secondes:02d}")
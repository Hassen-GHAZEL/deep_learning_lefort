from datetime import datetime, timedelta
from Shallow_network import *

# Obtenir l'heure de début
heure_de_debut = datetime.now().strftime("%H:%M:%S")

# Définition des valeurs à tester pour chaque hyperparamètre
tab_batch_size = list(range(1, 21, 2))
tab_nb_epochs = list(range(1, 21, 2))
tab_learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
tab_hidden_size = [32, 64, 128, 256, 512]
tab_weight_init_range = [(0, 0.1), (-0.1, 0.1), (-0.01, 0.01), (-0.001, 0.001), (-0.0001, 0.0001)]


nb_operation = len(tab_batch_size) * len(tab_learning_rate) * len(tab_hidden_size) * len(tab_weight_init_range)
nb_operation += sum(tab_nb_epochs)


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

print("INFLUENCE DE BATCH_SIZE (taille des lots de données pour l'entraînement)")


for batch_size in tab_batch_size:
    print("BATCH_SIZE : ", batch_size)
    # Chargement des hyperparamètres
    params = definir_hyperparametres(batch_size=batch_size)

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'], params['weight_init_range'])

    # Chargement des données
    train_loader, val_loader = charger_donnees(params)

    # Entraînement du modèle
    model.train_and_evaluate(train_loader, val_loader,  params, nb_operation, "batch_size")
print("INFLUENCE DE NB_EPOCHS (nombre d'époques d'entraînement)")


for nb_epochs in tab_nb_epochs:
    print("NB_EPOCHS : ", nb_epochs)
    # Chargement des hyperparamètres
    params = definir_hyperparametres(nb_epochs=nb_epochs)

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'], params['weight_init_range'])

    # Chargement des données
    train_loader, val_loader = charger_donnees(params)

    # Entraînement du modèle
    model.train_and_evaluate(train_loader, val_loader, params, nb_operation, "nb_epochs")

print("INFLUENCE DE LEARNING_RATE (taux d'apprentissage pour l'optimiseur)")


for learning_rate in tab_learning_rate:
    print("LEARNING_RATE : ", learning_rate)
    # Chargement des hyperparamètres
    params = definir_hyperparametres(learning_rate=learning_rate)

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'], params['weight_init_range'])

    # Chargement des données
    train_loader, val_loader = charger_donnees(params)

    # Entraînement du modèle
    model.train_and_evaluate(train_loader, val_loader, params, nb_operation, "learning_rate")


print("INFLUENCE DE HIDDEN_SIZE (nombre de neurones dans la couche cachée)")

for hidden_size in tab_hidden_size:
    print("HIDDEN_SIZE : ", hidden_size)
    # Chargement des hyperparamètres
    params = definir_hyperparametres(hidden_size=hidden_size)

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'], params['weight_init_range'])

    # Chargement des données
    train_loader, val_loader = charger_donnees(params)

    # Entraînement du modèle
    model.train_and_evaluate(train_loader, val_loader, params, nb_operation, "hidden_size")


print("INFLUENCE DE WEIGHT_INIT_RANGE (plage d'initialisation des poids)")

for weight_init_range in tab_weight_init_range:
    print("WEIGHT_INIT_RANGE : ", weight_init_range)
    # Chargement des hyperparamètres
    params = definir_hyperparametres(weight_init_range=weight_init_range)

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'], params['weight_init_range'])

    # Chargement des données
    train_loader, val_loader = charger_donnees(params)

    # Entraînement du modèle
    model.train_and_evaluate(train_loader, val_loader, params, nb_operation, "weight_init_range")


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
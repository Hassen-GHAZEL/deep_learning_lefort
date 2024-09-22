import torch
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import GPUtil

def definir_hyperparametres(batch_size=5, nb_epochs=10, learning_rate=0.0001, input_size=784, hidden_size=128, output_size=10, weight_init_range=(-0.001, 0.001)):
    """
    Fonction pour définir les hyperparamètres du modèle.
    Renvoie un dictionnaire contenant les hyperparamètres.
    """
    params = {
        'batch_size': batch_size,          # Taille des lots de données pour l'entraînement
        'nb_epochs': nb_epochs,            # Nombre d'époques d'entraînement
        'learning_rate': learning_rate,    # Taux d'apprentissage pour l'optimiseur
        'input_size': input_size,          # Nombre de caractéristiques d'entrée (28x28 pixels pour MNIST)
        'hidden_size': hidden_size,        # Nombre de neurones dans la couche cachée
        'output_size': output_size,        # Nombre de classes (0 à 9 pour MNIST)
        'weight_init_range': weight_init_range  # Plage d'initialisation des poids
    }
    return params

def check_list_type(variable):
    if isinstance(variable, list):  # Vérifie si c'est une liste
        if all(isinstance(i, list) for i in variable):  # Vérifie si tous les éléments sont des listes
            return "C'est une liste de listes"
        return "C'est une liste"
    return "Ce n'est pas une liste"


def charger_donnees(train_dataset, val_dataset, params):
    """
    Fonction pour charger les données et créer les DataLoader.
    """
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    return train_loader, val_loader


def calculer_ecart_temps(t1: str, t2: str) -> str:
    """
    Cette fonction calcule l'écart de temps entre deux horaires donnés sous forme de chaînes de caractères.
    """
    format_heure = "%H:%M:%S"
    t1_datetime = datetime.strptime(t1, format_heure)
    t2_datetime = datetime.strptime(t2, format_heure)

    if t2_datetime < t1_datetime:
        t2_datetime += timedelta(days=1)

    ecart = t2_datetime - t1_datetime
    heures, reste = divmod(ecart.seconds, 3600)
    minutes, secondes = divmod(reste, 60)

    return f"{heures:02d}:{minutes:02d}:{secondes:02d}"

def check_gpu():
    """
    Méthode statique pour vérifier si un GPU est disponible et obtenir son nom.
    """
    if torch.cuda.is_available():
        gpu_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_device)
        return f"GPU is available: {gpu_name}, currently using GPU: {gpu_device}"
    else:
        return "No GPU available, using CPU"

def create_or_overwrite_file(filename, content):
    # 'w' mode ouvre le fichier en mode écriture et l'écrase s'il existe déjà
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Le fichier '{filename}' a été créé/écrasé avec succès.")

def get_gpu_temperature():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None  # Aucun GPU trouvé
    for gpu in gpus:
        return gpu.temperature  # Retourne la température du premier GPU trouvé

def shutdown_system():
    """
    Méthode statique pour éteindre le système.
    """
    import os
    os.system("shutdown /s /t 1")
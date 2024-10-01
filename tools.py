import json
import time
import torch
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import GPUtil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess


def save_boxplot_with_stats(excel_file, sheet_name, column_name, output_file="boxplot.png"):
    # Charger les données depuis le fichier Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Vérifier que la colonne existe
    if column_name not in df.columns:
        raise ValueError(f"La colonne '{column_name}' n'existe pas dans la feuille '{sheet_name}'.")

    # Extraire les données de la colonne
    data = df[column_name].dropna()  # Supprime les valeurs NaN

    # Calcul des statistiques
    min_val = round(data.min(), 2)
    q1 = round(data.quantile(0.25), 2)
    median = round(data.median(), 2)
    q3 = round(data.quantile(0.75), 2)
    max_val = round(data.max(), 2)

    # Créer le boxplot sans les outliers (showfliers=False)
    plt.figure(figsize=(12, 6))  # Augmenter la taille de la figure
    ax = sns.boxplot(data=data, orient="h", showfliers=False)  # Désactive l'affichage des outliers

    # Ajouter les valeurs statistiques sous le boxplot
    stats = [min_val, q1, median, q3, max_val]

    # Position y pour le texte (au milieu de la boîte)
    y_pos = 0  # Pour un boxplot horizontal, la position y sera 0

    # Position x pour chaque annotation
    x_positions = [min_val, q1, median, q3, max_val]

    # Affichage des valeurs
    for stat, x_pos in zip(stats, x_positions):
        plt.text(x_pos, y_pos, f'{stat:.2f}', va='bottom', ha='center', color='black', fontsize=10)

    # Ajouter des titres et étiquettes
    plt.title(f"Box Plot de la colonne '{column_name}'", fontsize=14)
    plt.xlabel(column_name, fontsize=12)

    # Ajuster les marges
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    # Sauvegarder le plot en fichier PNG
    plt.savefig(output_file)
    plt.close()

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


import os

import getpass


def enregistrer_debut_programme(pid=None, filename="programme_log.txt", json_filename="programme_pid.json"):
    """
    Enregistre l'heure de démarrage du programme avec son PID dans un fichier texte et dans un fichier JSON.
    Si le fichier JSON existe déjà, il est écrasé.
    """
    if pid is None:
        pid = os.getpid()  # Obtenir le PID du processus en cours
    else:
        pid = int(pid)  # Convertir en int si c'est une chaîne

    # Récupérer l'heure actuelle
    debut = time.strftime("%H:%M:%S")

    # Vérifier si le fichier texte existe
    file_exists = os.path.isfile(filename)

    # Enregistrer dans le fichier texte
    with open(filename, 'a') as file:
        if file_exists:
            file.write("\n")
        file.write(f"programme pid={pid} démarré à {debut}\n")

    print(f"programme pid={pid} démarré à {debut}")

    # Sauvegarder le PID dans le fichier JSON
    with open(json_filename, 'w') as json_file:
        json.dump({'pid': pid}, json_file)
        print(f"Le PID {pid} a été enregistré dans '{json_filename}'.")


def enregistrer_fin_programme(pid=None, filename="programme_log.txt"):
    """
    Lit la dernière ligne du fichier et enregistre l'heure de fin du programme.
    Met à jour le fichier avec la durée d'exécution.
    """
    if pid is None:
        pid = os.getpid()  # Obtenir le PID du processus en cours

    fin = time.strftime("%H:%M:%S")
    debut = ""
    derniere_ligne = ""

    # Lire la dernière ligne du fichier
    try:
        with open(filename, 'r') as file:
            lignes = file.readlines()
            derniere_ligne = lignes[-1].strip()  # Dernière ligne
            debut = derniere_ligne.split("démarré à ")[1]  # Extraire l'heure de début
    except (FileNotFoundError, IndexError):
        print(f"Aucun enregistrement trouvé dans '{filename}'.")
        return

    # Calculer la durée
    duree = calculer_ecart_temps(debut, fin)

    # Afficher le message
    print(f"{derniere_ligne}, fini à {fin}, durée {duree}")

    # Mettre à jour la dernière ligne dans le fichier
    with open(filename, 'a') as file:
        file.write(f", fini à {fin}, durée {duree}\n")

    # Arrêter le programme
    os.system(f"taskkill /PID {pid} /F")

def lire_pid_du_fichier(json_filename="programme_pid.json")->int:
    """
    Lit le fichier JSON et retourne le PID enregistré en tant qu'entier.

    :param json_filename: Le nom du fichier JSON à lire.
    :return: PID en tant qu'entier ou None si le fichier n'existe pas ou si le PID n'est pas valide.
    """
    try:
        with open(json_filename, 'r') as json_file:
            data = json.load(json_file)  # Charger le contenu JSON
            pid = data.get('pid')  # Récupérer la valeur du PID
            return int(pid)  # Convertir et retourner le PID en tant qu'entier
    except FileNotFoundError:
        print(f"Le fichier '{json_filename}' n'a pas été trouvé.")
        return -10
    except (ValueError, TypeError):
        print("Le PID dans le fichier JSON n'est pas valide.")
        return -10

def git_commit_and_push(commit_message: str):
    """
    Exécute les commandes Git pour ajouter, committer et pousser les modifications.

    :param commit_message: Le message de commit à utiliser pour le commit Git.
    """
    try:
        # Exécuter la commande 'git add .'
        add_result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True, check=True)
        print("Résultat de 'git add .':")
        print(add_result.stdout)

        # Exécuter la commande 'git commit -m commit_message'
        commit_result = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True, check=True)
        print("Résultat de 'git commit':")
        print(commit_result.stdout)

        # Exécuter la commande 'git push'
        push_result = subprocess.run(['git', 'push'], capture_output=True, text=True, check=True)
        print("Résultat de 'git push':")
        print(push_result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Une erreur s'est produite lors de l'exécution de Git : {e.stderr}")

def shutdown_system():
    """
    Méthode statique pour éteindre le système de force.
    """
    import os
    os.system("shutdown /s /f /t 1")

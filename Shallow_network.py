import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gzip  # Assurez-vous d'importer le module gzip
import numpy as np
from Excel import *

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

# Définition de la classe pour le modèle de réseau de neurones
class PerceptronMulticouche(nn.Module):

    count = 0
    column_name = ["batch_size", "nb_epochs", "learning_rate", "input_size", "hidden_size", "output_size", "weight_init_range", "train_loss", "val_loss", "accuracy"]
    column_name = column_name + ["Train Loss", "Val Loss", "Accuracy"]
    excel = ExcelManager("tableau1.xlsx", definir_hyperparametres().keys())
    def __init__(self, input_size, hidden_size, output_size, weight_init_range):
        super(PerceptronMulticouche, self).__init__()

        # Définition des couches du réseau
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Initialisation des poids
        nn.init.uniform_(self.hidden.weight, *weight_init_range)
        nn.init.uniform_(self.output.weight, *weight_init_range)

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Fonction d'activation ReLU pour la couche cachée
        x = self.output(x)  # Sortie linéaire
        return x

    def train_and_evaluate(self, train_loader, val_loader, params, total_call, sheet_name):
        # Définir l'optimiseur et la fonction de perte
        optimizer = optim.SGD(self.parameters(), lr=params['learning_rate'])
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(params['nb_epochs']):
            # Phase d'entraînement
            self.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                loss = loss_func(y_pred, torch.argmax(y_batch, dim=1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Phase de validation
            self.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    y_val_pred = self.forward(x_val)
                    loss = loss_func(y_val_pred, torch.argmax(y_val, dim=1))
                    val_loss += loss.item()
                    _, predicted = torch.max(y_val_pred, 1)
                    correct += (predicted == torch.argmax(y_val, dim=1)).sum().item()
                    total += y_val.size(0)

            val_loss /= len(val_loader)
            accuracy = correct * 100 / total

            # Affichage des métriques

            PerceptronMulticouche.count += 1
            print(f"PerceptronMulticouche.count/total_call  : {PerceptronMulticouche.count}/{total_call} = {(PerceptronMulticouche.count*100/total_call):.3f}%")
            print(f"Epoch {epoch + 1}/{params['nb_epochs']}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Accuracy: {accuracy :.4f}%")
            if epoch + 1 == params['nb_epochs']:
                row = [valeur for valeur in params.values()]
                row = row + [train_loss, val_loss, accuracy]
                PerceptronMulticouche.excel.add_row(sheet_name, row)
"""
# Exemple d'utilisation
if __name__ == "__main__":
    # Chargement des hyperparamètres
    params = definir_hyperparametres()

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'])

    # Chargement des données
    model.charger_donnees('mnist.pkl.gz', params)

    # Entraînement du modèle
    model.train_mlp(params)
"""
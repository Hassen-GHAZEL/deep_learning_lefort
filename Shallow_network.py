import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tools import calculer_ecart_temps


class PerceptronMulticouche(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, weight_init_range, excel, gpu_automatic=True):
        """Initialise le perceptron multicouche avec les dimensions des entrées, sorties, la plage d'initialisation des poids et un objet Excel."""
        super(PerceptronMulticouche, self).__init__()

        self.excel = excel

        # Détection du GPU et définition du device (GPU si disponible, sinon CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu_automatic else "cpu")

        # Définition des couches du réseau
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Initialisation des poids
        nn.init.uniform_(self.hidden.weight, *weight_init_range)
        nn.init.uniform_(self.output.weight, *weight_init_range)

        # Déplacer le modèle sur le device (GPU ou CPU)
        self.to(self.device)

    def forward(self, x):
        """
        Fonction de propagation avant du réseau.
        Le tenseur d'entrée est déplacé vers le bon appareil (device).
        """
        # Déplacer l'entrée sur le bon device (CPU ou GPU)
        x = x.to(self.device)
        x = torch.relu(self.hidden(x))  # Fonction d'activation ReLU pour la couche cachée
        x = self.output(x)  # Sortie linéaire
        return x

    def train_and_evaluate(self, sheet_name, train_loader, val_loader, test_loader, params, is_nested=True):
        """Entraîne et évalue le modèle sur les ensembles de données fournis."""
        optimizer = optim.SGD(self.parameters(), lr=params['learning_rate'])  # Optimiseur SGD
        loss_func = nn.CrossEntropyLoss()  # Fonction de perte
        rows = []

        for epoch in range(params['nb_epochs']):
            debut_iteration = datetime.now().strftime("%H:%M:%S")
            if not is_nested:
                print(f"\t\tEpoch {epoch + 1}/{params['nb_epochs']} : {debut_iteration}")

            # Phase d'entraînement
            self.train()  # Met le modèle en mode entraînement
            train_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()  # Réinitialise les gradients

                # Déplacer les données sur le bon device
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                y_pred = self.forward(x_batch)  # Prédiction
                loss = loss_func(y_pred, torch.argmax(y_batch, dim=1))  # Calcul de la perte
                loss.backward()  # Rétropropagation
                optimizer.step()  # Mise à jour des poids
                train_loss += loss.item()

            train_loss /= len(train_loader)  # Moyenne de la perte d'entraînement

            # Phase de validation
            self.eval()  # Met le modèle en mode évaluation
            val_loss = 0
            with torch.no_grad():  # Désactive le calcul des gradients
                for x_val, y_val in val_loader:
                    # Déplacer les données sur le bon device
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)

                    y_val_pred = self.forward(x_val)  # Prédiction sur les données de validation
                    loss = loss_func(y_val_pred, torch.argmax(y_val, dim=1))  # Calcul de la perte
                    val_loss += loss.item()

            val_loss /= len(val_loader)  # Moyenne de la perte de validation

            # Phase de test
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    # Déplacer les données sur le bon device
                    x_test, y_test = x_test.to(self.device), y_test.to(self.device)

                    y_test_pred = self.forward(x_test)  # Prédiction sur les données de test
                    loss = loss_func(y_test_pred, torch.argmax(y_test, dim=1))  # Calcul de la perte
                    test_loss += loss.item()

                    # Calcul de la précision
                    _, predicted = torch.max(y_test_pred, 1)
                    correct += (predicted == torch.argmax(y_test, dim=1)).sum().item()
                    total += y_test.size(0)

            test_loss /= len(test_loader)  # Moyenne de la perte de test
            accuracy = correct * 100 / total  # Calcul de la précision

            # Affichage et enregistrement des résultats
            if not is_nested:
                print(f"\t\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Duration: "
                      f"{calculer_ecart_temps(debut_iteration, datetime.now().strftime('%H:%M:%S'))}")
            rows.append([epoch + 1] + list(params.values()) + [train_loss, val_loss, test_loss, accuracy])

        for row in rows:
            self.excel.add_row(sheet_name, row)  # Ajout des résultats à l'Excel

        # Pour les appels imbriqués (ex. hyperparameter tuning), enregistrement succinct
        if is_nested:
            print(f"\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        else:
            self.excel.add_row(sheet_name, self.excel.column_titles)  # Enregistrement des en-têtes si pas imbriqué

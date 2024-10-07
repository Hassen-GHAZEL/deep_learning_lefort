import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tools import calculer_ecart_temps


class DeepNetwork(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, weight_init_range, excel, gpu_automatic = True):
        """
        input_size : taille des entrées
        hidden_layers : liste d'entiers représentant le nombre de neurones dans chaque couche cachée
        output_size : taille des sorties
        weight_init_range : plage pour l'initialisation des poids
        excel : objet Excel pour enregistrer les résultats
        """
        super(DeepNetwork, self).__init__()

        self.excel = excel

        # Détection automatique du GPU. Si disponible, use_gpu est défini à True.
        self.use_gpu = torch.cuda.is_available() and gpu_automatic

        # Supprimer les couches avec 0 neurones
        hidden_layers = [h for h in hidden_layers if h > 0]

        # Vérifier qu'il y a au moins deux couches cachées
        if len(hidden_layers) < 2:
            raise ValueError("Le réseau doit contenir au moins deux couches cachées valides.")

        # Définir les couches du réseau
        self.layers = nn.ModuleList()
        previous_size = input_size
        
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(previous_size, hidden_size))
            previous_size = hidden_size

        # Ajouter la couche de sortie après les couches cachées
        self.output_layer = nn.Linear(previous_size, output_size)

        # Initialisation des poids pour chaque couche
        for layer in self.layers:
            nn.init.uniform_(layer.weight, *weight_init_range)
        nn.init.uniform_(self.output_layer.weight, *weight_init_range)

        # Si GPU est disponible, déplacer le modèle sur GPU
        if self.use_gpu:
            self.cuda()

    def forward(self, x):
        """
        Fonction de propagation avant du réseau.
        Si un GPU est disponible, les tenseurs sont déplacés sur le GPU.
        """
        if self.use_gpu:
            x = x.cuda()  # Déplacer le tenseur sur GPU si nécessaire
        
        # Propagation à travers les couches cachées avec une activation ReLU
        for layer in self.layers:
            x = torch.relu(layer(x))
        
        # Sortie linéaire
        x = self.output_layer(x)
        return x

    def train_and_evaluate(self, sheet_name, train_loader, val_loader, test_loader, params, is_nested=True):
        optimizer = optim.SGD(self.parameters(), lr=params['learning_rate'])
        loss_func = nn.CrossEntropyLoss()
        rows = []
        for epoch in range(params['nb_epochs']):
            debut_iteration = datetime.now().strftime("%H:%M:%S")
            if not is_nested:
                print(f"\t\tEpoch {epoch + 1}/{params['nb_epochs']} : {debut_iteration}")

            # Phase d'entraînement
            self.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # Déplacer les données sur GPU si nécessaire
                if self.use_gpu:
                    x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

                y_pred = self.forward(x_batch)
                loss = loss_func(y_pred, torch.argmax(y_batch, dim=1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Phase de validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    if self.use_gpu:
                        x_val, y_val = x_val.cuda(), y_val.cuda()

                    y_val_pred = self.forward(x_val)
                    loss = loss_func(y_val_pred, torch.argmax(y_val, dim=1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Phase de test
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    if self.use_gpu:
                        x_test, y_test = x_test.cuda(), y_test.cuda()

                    y_test_pred = self.forward(x_test)
                    loss = loss_func(y_test_pred, torch.argmax(y_test, dim=1))
                    test_loss += loss.item()

                    _, predicted = torch.max(y_test_pred, 1)
                    correct += (predicted == torch.argmax(y_test, dim=1)).sum().item()
                    total += y_test.size(0)

            test_loss /= len(test_loader)
            accuracy = correct * 100 / total

            # Affichage et enregistrement des résultats
            if not is_nested:
                print(f"\t\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Duration: "
                      f"{calculer_ecart_temps(debut_iteration, datetime.now().strftime('%H:%M:%S'))}")
            rows.append([epoch + 1] + list(params.values()) + [train_loss, val_loss, test_loss, accuracy])

        for row in rows:
            self.excel.add_row(sheet_name, row)

        # Pour les appels imbriqués (ex. hyperparameter tuning), enregistrement succinct
        if is_nested:
            print(
                f"\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        else:
            self.excel.add_row(sheet_name, self.excel.column_titles)

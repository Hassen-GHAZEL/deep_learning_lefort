import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tools import calculer_ecart_temps

class PerceptronMulticouche(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, weight_init_range, excel):
        super(PerceptronMulticouche, self).__init__()

        self.excel = excel

        # Détection automatique du GPU. Si disponible, use_gpu est défini à True.
        self.use_gpu = torch.cuda.is_available()

        # Définition des couches du réseau
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Initialisation des poids
        nn.init.uniform_(self.hidden.weight, *weight_init_range)
        nn.init.uniform_(self.output.weight, *weight_init_range)

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
        x = torch.relu(self.hidden(x))  # Fonction d'activation ReLU pour la couche cachée
        x = self.output(x)  # Sortie linéaire
        return x

    def train_and_evaluate(self, sheet_name, train_loader, val_loader, params, is_nested = True):
        optimizer = optim.SGD(self.parameters(), lr=params['learning_rate'])
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(params['nb_epochs']):
            debut_iteration = datetime.now().strftime("%H:%M:%S")
            if not is_nested:
                print(f"\t\tEpoch {epoch + 1}/{params['nb_epochs']} : {debut_iteration}")
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

            self.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    if self.use_gpu:
                        x_val, y_val = x_val.cuda(), y_val.cuda()

                    y_val_pred = self.forward(x_val)
                    loss = loss_func(y_val_pred, torch.argmax(y_val, dim=1))
                    val_loss += loss.item()

                    _, predicted = torch.max(y_val_pred, 1)
                    correct += (predicted == torch.argmax(y_val, dim=1)).sum().item()
                    total += y_val.size(0)

            val_loss /= len(val_loader)
            accuracy = correct * 100 / total

            if not is_nested:
                print(f"\t\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy}, Duration: {calculer_ecart_temps(debut_iteration, datetime.now().strftime('%H:%M:%S'))}")
                self.excel.add_row(sheet_name, [epoch + 1] + list(params.values()) + [train_loss, val_loss, accuracy])

        if is_nested:
            print(f"\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
            self.excel.add_row(sheet_name, list(params.values()) + [train_loss, val_loss, accuracy])
        else:
            self.excel.add_row(sheet_name, self.excel.column_titles)


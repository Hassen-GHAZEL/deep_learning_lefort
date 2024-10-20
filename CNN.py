from datetime import datetime
from tools import calculer_ecart_temps
import torch

class MNISTModel:
    def __init__(self, num_classes=10, batch_size=5, learning_rate=0.001, nb_epochs=10, excel=None, gpu_automatic=True):
        # Initialisation des paramètres du modèle
        self.num_classes = num_classes  # Nombre de classes de sortie (10 pour MNIST)
        self.batch_size = batch_size  # Taille des lots pour l'entraînement
        self.learning_rate = learning_rate  # Taux d'apprentissage pour l'optimiseur
        self.nb_epochs = nb_epochs  # Nombre d'époques pour l'entraînement
        self.excel = excel  # Objet pour gérer les résultats dans un fichier Excel
        self.device = torch.device('cuda' if gpu_automatic and torch.cuda.is_available() else 'cpu')  # Définir l'appareil (CPU ou GPU)
        self.model = self.build_model()  # Construire le modèle
        self.loss_func = torch.nn.CrossEntropyLoss()  # Fonction de perte (CrossEntropyLoss)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)  # Optimiseur (SGD)

    def build_model(self):
        # Construction de l'architecture du modèle
        kernel_size = 5  # Taille du noyau pour les couches de convolution
        pooling_params = (2, 2)  # Paramètres de la couche de pooling
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=kernel_size),  # Couche de convolution
            torch.nn.BatchNorm2d(6),  # Normalisation par lots
            torch.nn.MaxPool2d(kernel_size=pooling_params[0], stride=pooling_params[1]),  # Couche de pooling
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size),  # Deuxième couche de convolution
            torch.nn.BatchNorm2d(16),  # Normalisation par lots
            torch.nn.MaxPool2d(kernel_size=pooling_params[0], stride=pooling_params[1]),  # Deuxième couche de pooling
            torch.nn.Flatten(),  # Aplatir les données pour les couches linéaires
            torch.nn.Linear(16 * 4 * 4, 120),  # Couche linéaire avec 120 neurones
            torch.nn.ReLU(),  # Fonction d'activation ReLU
            torch.nn.Linear(120, 84),  # Couche linéaire avec 84 neurones
            torch.nn.ReLU(),  # Fonction d'activation ReLU
            torch.nn.Linear(84, self.num_classes)  # Couche de sortie avec un neurone par classe
        )
        return model.to(self.device)  # Transférer le modèle sur l'appareil défini

    def train_and_evaluate(self, sheet_name, train_loader, val_loader, test_loader, is_nested=True):
        # Entraîner et évaluer le modèle
        rows = []  # Liste pour stocker les résultats des époques
        for epoch in range(self.nb_epochs):
            debut_iteration = datetime.now().strftime("%H:%M:%S")  # Heure de début de l'itération
            if not is_nested:
                print(f"\t\tEpoch {epoch + 1}/{self.nb_epochs} : {debut_iteration}")
            self.model.train()  # Mettre le modèle en mode entraînement
            train_loss = 0  # Initialiser la perte d'entraînement
            correct_predictions = 0  # Compteur pour les prédictions correctes
            total_samples = 0  # Compteur pour le nombre total d'échantillons

            # Boucle d'entraînement sur les batches
            for x, t in train_loader:
                x, t = x.to(self.device), t.to(self.device)  # Transférer les données sur l'appareil
                x = x.view(-1, 1, 28, 28)  # Redimensionnement dynamique des entrées
                y = self.model(x)  # Passer les données à travers le modèle
                loss = self.loss_func(y, t)  # Calculer la perte
                self.optim.zero_grad()  # Réinitialiser les gradients
                loss.backward()  # Calculer les gradients
                self.optim.step()  # Mettre à jour les poids

                train_loss += loss.item()  # Ajouter la perte d'entraînement

                # Calculer l'accuracy pour le batch
                predicted_classes = torch.argmax(y, 1)  # Prédictions du modèle
                correct_predictions += (predicted_classes == torch.argmax(t, 1)).sum().item()  # Compter les bonnes prédictions
                total_samples += t.size(0)  # Nombre d'exemples dans le batch

            train_loss /= len(train_loader)  # Moyenne de la perte d'entraînement
            train_accuracy = correct_predictions / total_samples  # Calculer l'accuracy

            # Évaluer sur le jeu de validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Évaluer sur le jeu de test
            test_loss, test_acc = self.evaluate(test_loader)
            test_acc *= 100  # Convertir en pourcentage

            # Affichage et enregistrement des résultats
            if not is_nested:
                print(f"\t\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Duration: "
                      f"{calculer_ecart_temps(debut_iteration, datetime.now().strftime('%H:%M:%S'))}")
            # Ajouter les résultats de l'époque à la liste
            rows.append([epoch + 1, self.nb_epochs] + [self.batch_size, self.learning_rate] + [train_loss, val_loss, test_loss, test_acc])

        # Enregistrer les résultats dans le fichier Excel
        for row in rows:
            self.excel.add_row(sheet_name, row)

        # Pour les appels imbriqués (ex. tuning des hyperparamètres), enregistrement succinct
        if is_nested:
            print(
                f"\tTraining Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    def evaluate(self, loader):
        # Évaluer le modèle sur un ensemble de données
        self.model.eval()  # Mettre le modèle en mode évaluation
        loss = 0  # Initialiser la perte
        correct = 0  # Compteur pour les prédictions correctes
        with torch.no_grad():  # Pas de calcul des gradients lors de l'évaluation
            for x, t in loader:
                x, t = x.to(self.device), t.to(self.device)  # Transférer les données sur l'appareil
                x = torch.reshape(x, (x.size(0), 1, 28, 28))  # Redimensionner les données
                y = self.model(x)  # Passer les données à travers le modèle
                loss += self.loss_func(y, t).item()  # Calculer la perte
                correct += (torch.argmax(y, 1) == torch.argmax(t, 1)).sum().item()  # Compter les bonnes prédictions
        loss /= len(loader)  # Moyenne de la perte
        accuracy = correct / len(loader.dataset)  # Calculer l'accuracy
        return loss, accuracy  # Retourner la perte et l'accuracy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datetime import datetime
from tools import calculer_ecart_temps

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, train_loader, val_loader, test_loader, learning_rate=0.001):
        """
        num_classes : nombre de classes pour la classification
        train_loader : DataLoader pour les données d'entraînement
        val_loader : DataLoader pour les données de validation
        test_loader : DataLoader pour les données de test
        learning_rate : taux d'apprentissage pour l'optimiseur
        """
        super(ResNetClassifier, self).__init__()
        
        # Charger le modèle ResNet pré-entraîné
        self.model = models.resnet18(pretrained=True)
        
        # Modifier la dernière couche pour correspondre au nombre de classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Initialiser l'optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Enregistrer les chargeurs de données
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def forward(self, x):
        return self.model(x)

    def train_and_evaluate(self, num_epochs):
        rows = []

        for epoch in range(num_epochs):
            debut_iteration = datetime.now().strftime("%H:%M:%S")
            print(f"Epoch {epoch + 1}/{num_epochs} : {debut_iteration}")

            # Phase d'entraînement
            self.model.train()
            train_loss = 0
            for x_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                
                # Déplacer les données sur le GPU si disponible
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Forward pass
                y_pred = self.forward(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            # Phase de validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for x_val, y_val in self.val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    y_val_pred = self.forward(x_val)
                    loss = self.loss_fn(y_val_pred, y_val)
                    val_loss += loss.item()

                    _, predicted = torch.max(y_val_pred, 1)
                    correct += (predicted == y_val).sum().item()
                    total += y_val.size(0)

            val_loss /= len(self.val_loader)
            val_accuracy = correct / total * 100

            # Phase de test
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for x_test, y_test in self.test_loader:
                    x_test, y_test = x_test.to(device), y_test.to(device)

                    y_test_pred = self.forward(x_test)
                    loss = self.loss_fn(y_test_pred, y_test)
                    test_loss += loss.item()

                    _, predicted = torch.max(y_test_pred, 1)
                    correct += (predicted == y_test).sum().item()
                    total += y_test.size(0)

            test_loss /= len(self.test_loader)
            test_accuracy = correct / total * 100

            # Affichage des résultats
            print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.2f}%, Test Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy:.2f}%, Duration: "
                  f"{calculer_ecart_temps(debut_iteration, datetime.now().strftime('%H:%M:%S'))}")

            # Enregistrer les résultats
            rows.append([epoch + 1, train_loss, val_loss, val_accuracy, test_loss, test_accuracy])

        return rows


# Exemple d'utilisation
if __name__ == "__main__":
    # Créez vos DataLoader pour l'entraînement, la validation et le test ici
    # train_loader, val_loader, test_loader = ...

    # Créez le modèle
    num_classes = 10  # Par exemple, pour un problème de classification à 10 classes
    model = ResNetClassifier(num_classes, train_loader, val_loader, test_loader)

    # Définir le nombre d'époques
    num_epochs = 10

    # Entraîner et évaluer le modèle
    results = model.train_and_evaluate(num_epochs)

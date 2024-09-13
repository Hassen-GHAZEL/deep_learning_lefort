import gzip
import torch
from torch.utils.data import DataLoader, TensorDataset

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
class PerceptronMulticouche(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PerceptronMulticouche, self).__init__()
        # Définition des couches du modèle
        self.hidden_layer = torch.nn.Linear(input_size, hidden_size)  # Couche cachée
        self.output_layer = torch.nn.Linear(hidden_size, output_size) # Couche de sortie
        self.activation = torch.nn.ReLU()  # Fonction d'activation ReLU
        self.train_loader = None  # Chargeur de données d'entraînement (sera initialisé plus tard)
        self.test_loader = None   # Chargeur de données de test (sera initialisé plus tard)

    def forward(self, x):
        # Calcul du passage avant (forward) à travers le réseau
        hidden_output = self.activation(self.hidden_layer(x))  # Sortie de la couche cachée
        output = self.output_layer(hidden_output)  # Sortie de la couche de sortie
        return output

    def charger_donnees(self, chemin_fichier, params):
        """
        Méthode pour charger les données MNIST et créer les ensembles de données et les chargeurs de données.
        """
        # Chargement des données MNIST depuis un fichier compressé
        with gzip.open(chemin_fichier, 'rb') as f:
            (data_train, label_train), (data_test, label_test) = torch.load(f)

        # Création des ensembles de données de formation et de test
        train_dataset = TensorDataset(data_train, label_train)
        test_dataset = TensorDataset(data_test, label_test)

        # Création des chargeurs de données (DataLoader) pour l'entraînement et le test
        self.train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def train_mlp(self, params):
        """
        Méthode pour entraîner le perceptron multicouche et afficher les pertes d'entraînement et de validation.
        """
        # Initialisation aléatoire des poids du modèle avec les hyperparamètres définis
        weight_init_min, weight_init_max = params['weight_init_range']
        torch.nn.init.uniform_(self.hidden_layer.weight, weight_init_min, weight_init_max)
        torch.nn.init.uniform_(self.output_layer.weight, weight_init_min, weight_init_max)

        # Définition de la fonction de perte et de l'optimiseur
        loss_func = torch.nn.CrossEntropyLoss()  # Fonction de perte pour la classification multi-classe
        optimizer = torch.optim.SGD(self.parameters(), lr=params['learning_rate'])  # Optimiseur SGD avec le taux d'apprentissage défini

        # Boucle d'entraînement sur le nombre d'époques spécifié
        for epoch in range(params['nb_epochs']):
            self.train()  # Met le modèle en mode entraînement
            total_loss = 0  # Initialisation de la perte totale pour cette époque

            # Itération sur les lots de données d'entraînement
            for x, t in self.train_loader:
                optimizer.zero_grad()  # Réinitialisation des gradients
                y = self(x)  # Calcul de la sortie du modèle
                loss = loss_func(y, t)  # Calcul de la perte
                loss.backward()  # Calcul des gradients pour la rétropropagation
                optimizer.step()  # Mise à jour des poids du modèle
                total_loss += loss.item()  # Accumulation de la perte pour affichage

            # Calcul de la perte de validation
            self.eval()  # Met le modèle en mode évaluation
            val_loss = 0  # Initialisation de la perte totale pour la validation
            correct = 0  # Compteur pour les prédictions correctes
            total = 0  # Nombre total d'exemples de test
            with torch.no_grad():  # Désactive la dérivation des gradients pour l'évaluation
                for x, t in self.test_loader:
                    y = self(x)  # Calcul de la sortie du modèle
                    loss = loss_func(y, t)  # Calcul de la perte de validation
                    val_loss += loss.item()  # Accumulation de la perte de validation
                    
                    pred = torch.argmax(y, dim=1)  # Prédiction de la classe la plus probable
                    correct += (pred == t).sum().item()  # Incrémente le compteur si la prédiction est correcte
                    
                    total += t.size(0)  # Incrémente le nombre total d'exemples

            # Calcul de la précision et de la perte de validation moyenne
            accuracy = 100 * correct / total if total > 0 else 0  # Calcul de la précision en pourcentage
            avg_train_loss = total_loss / len(self.train_loader)  # Perte d'entraînement moyenne
            avg_val_loss = val_loss / len(self.test_loader)  # Perte de validation moyenne

            # Affichage des résultats
            print(f'Epoch {epoch+1}/{params["nb_epochs"]}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')


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

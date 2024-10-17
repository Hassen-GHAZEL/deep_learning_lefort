import gzip  # Assurez-vous d'importer le module gzip
from Deep_network import * # Assurez-vous que cette classe est correctement définie
from torch.utils.data import TensorDataset, random_split
from Excel import ExcelManager
from tools import *
from constantes import *
from datetime import datetime

def evaluer_hyperparametre(nom, valeurs):
    global excel
    print(f"Influence de {nom} sur le modèle :")
    for valeur in valeurs:
        heure_debut_iteration = datetime.now().strftime("%H:%M:%S")
        print(f"\t{nom} : {valeur}")

        params = definir_hyperparametres(**{nom.lower(): valeur})
        print(f"\tHyperparamètres : {params}")

        model = SimpleResNet()  # Initialisation du modèle

        # Charger les données
        train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params['batch_size'])

        # Entraînement et évaluation
        train_and_evaluate(model, train_loader, val_loader, test_loader, params)

        heure_fin_iteration = datetime.now().strftime("%H:%M:%S")
        ecart = calculer_ecart_temps(heure_debut_iteration, heure_fin_iteration)
        print(f"\tDurée de cette itération ({nom}={valeur}): {ecart}")

def charger_donnees(train_dataset, test_dataset, batch_size):
    """
    Charger et préparer les jeux de données avec la validation.
    """
    validation_split = 0.2
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_and_evaluate(model, train_loader, val_loader, test_loader, params):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Phase d'entraînement
    model.train()
    for epoch in range(params['nb_epochs']):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Phase de validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")


if __name__ == '__main__':
    # Obtenir l'heure de début
    enregistrer_debut_programme()

    # Chargement des données
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        (data_train, label_train), (data_test, label_test) = torch.load(f)

    # Préparation des jeux de données
    train_dataset = TensorDataset(data_train.view(-1, 1, 28, 28), label_train)  # Redimensionner pour 28x28
    test_dataset = TensorDataset(data_test.view(-1, 1, 28, 28), label_test)  # Redimensionner pour 28x28

    # Noms des colonnes pour Excel
    column_names = ["numéro epoch"] + list(definir_hyperparametres().keys()) + ["Training Loss", "Validation Loss", "Test Loss", "Accuracy"]

    # Initialisation de la gestion du fichier Excel
    excel = ExcelManager("excel/deep_network.xlsx", column_names)

    # Définition des valeurs à tester pour chaque hyperparamètre
    evaluer_hyperparametre("BATCH_SIZE", tab_batch_size)
    evaluer_hyperparametre("LEARNING_RATE", tab_learning_rate)
    evaluer_hyperparametre("HIDDEN_SIZE", tab_hidden_size)
    evaluer_hyperparametre("WEIGHT_INIT_RANGE", tab_weight_init_range)

    # Calculer et afficher le temps total d'exécution
    enregistrer_fin_programme()
    git_commit_and_push("Deep Network, analyse hyperparamètres CNN terminée")
import torch
import gzip
from torch.utils.data import TensorDataset
from CNN import MNISTModel
from tools import charger_donnees, definir_hyperparametres
from Excel import ExcelManager

def main():
    # Charger les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('data/mnist.pkl.gz'), map_location='cpu')

    # Créer des ensembles de données
    train_dataset = TensorDataset(data_train, label_train)
    test_dataset = TensorDataset(data_test, label_test)

    # Définir les paramètres
    params = definir_hyperparametres()

    print(f"Hyperparamètres : {params}")

    # Noms des colonnes pour Excel
    column_names = ["numéro epoch", "nb_epochs"] + ["batch_size", "learning_rate"] + ["Training Loss", "Validation Loss",
                                                                                "Test Loss", "Accuracy"]

    # Initialisation de la gestion du fichier Excel
    excel = ExcelManager("excel/CNN.xlsx", column_names)

    # Charger les données d'entraînement et de test avec validation
    train_loader, val_loader, test_loader = charger_donnees(train_dataset, test_dataset, params)

    # Initialiser le modèle
    model = MNISTModel(num_classes=params["output_size"], batch_size=params['batch_size'], learning_rate=params['learning_rate'], nb_epochs=params['nb_epochs'], excel=excel)

    # Entraîner et évaluer le modèle
    model.train_and_evaluate("Sheet1", train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()


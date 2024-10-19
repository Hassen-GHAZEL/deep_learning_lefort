import torch

class MNISTModel:
    def __init__(self, num_classes=10, batch_size=5, learning_rate=0.001, nb_epochs=10, excel=None, gpu_automatic=True):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.excel = excel
        self.device = torch.device('cuda' if gpu_automatic and torch.cuda.is_available() else 'cpu')
        self.model = self.build_model()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def build_model(self):
        kernel_size = 5
        pooling_params = (2, 2)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=kernel_size),
            torch.nn.BatchNorm2d(6),
            torch.nn.MaxPool2d(kernel_size=pooling_params[0], stride=pooling_params[1]),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(kernel_size=pooling_params[0], stride=pooling_params[1]),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, self.num_classes)
        )
        return model.to(self.device)

    def train_and_evaluate(self, train_loader, val_loader, test_loader):
        for epoch in range(self.nb_epochs):
            self.model.train()
            train_loss = 0
            correct_predictions = 0
            total_samples = 0

            for x, t in train_loader:
                x, t = x.to(self.device), t.to(self.device)
                x = x.view(-1, 1, 28, 28)  # Redimensionnement dynamique
                y = self.model(x)
                loss = self.loss_func(y, t)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_loss += loss.item()

                # Calculer l'accuracy pour le batch
                predicted_classes = torch.argmax(y, 1)
                correct_predictions += (predicted_classes == torch.argmax(t, 1)).sum().item()  # Modifié ici
                total_samples += t.size(0)  # Nombre d'exemples dans le batch

            train_loss /= len(train_loader)
            train_accuracy = correct_predictions / total_samples  # Calculer l'accuracy

            # Évaluer sur le jeu de validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Évaluer sur le jeu de test
            test_loss, test_acc = self.evaluate(test_loader)
            test_acc*=100

            print(
                f'Epoch {epoch + 1}/{self.nb_epochs}, Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')


    def evaluate(self, loader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for x, t in loader:
                x, t = x.to(self.device), t.to(self.device)
                x = torch.reshape(x, (x.size(0), 1, 28, 28))
                y = self.model(x)
                loss += self.loss_func(y, t).item()
                correct += (torch.argmax(y, 1) == torch.argmax(t, 1)).sum().item()  # Modifié ici
        loss /= len(loader)
        accuracy = correct / len(loader.dataset)
        return loss, accuracy


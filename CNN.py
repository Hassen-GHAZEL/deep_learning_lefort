# Définir un modèle CNN simple utilisant ResNet
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # C1: Convolution
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # S2: Pooling
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # C3: Convolution
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # C5: Fully Connected
        self.fc2 = nn.Linear(120, 84)  # F6: Fully Connected
        self.fc3 = nn.Linear(84, 10)  # Couche de Sortie pour 10 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # C1 + S2
        x = self.pool(self.relu(self.conv2(x)))  # C3 + S4
        x = x.view(-1, 16 * 5 * 5)  # Aplatissement
        x = self.relu(self.fc1(x))  # C5
        x = self.relu(self.fc2(x))  # F6
        x = self.fc3(x)  # Sortie
        return x

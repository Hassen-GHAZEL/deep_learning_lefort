import gzip, datetime, torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device(
# 	f'cuda:{torch.cuda.current_device()}'
# 	if torch.cuda.is_available()
# 	else 'cpu'
# )
# torch.set_default_device(device)

if __name__ == '__main__':
    print(f" programme lancé à {datetime.datetime.now().strftime('%H:%M:%S')}")
    batch_size = 5  # nombre de données lues à chaque fois
    print('batch size =', batch_size)
    nb_epochs = 10  # nombre de fois que la base de données sera lue
    eta = 0.001  # taux d'apprentissage
    print('eta =', eta)

    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('data/mnist.pkl.gz'),
                                                                      map_location=lambda storage, loc: storage.cuda(0))
    # on crée les lecteurs de données
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    input_size = data_train.shape[1]
    output_size = label_train.shape[1]
    init_range = 0.1
    print('init_range =', init_range)

    # on initialise le modèle et ses poids
    # model = torch.nn.Linear(data_train.shape[1],label_train.shape[1])
    kernel_size = 5
    pooling_params = (2, 2)
    picture_dim = 28
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=kernel_size),
        torch.nn.MaxPool2d(kernel_size=pooling_params[0], stride=pooling_params[1]),
        torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size),
        torch.nn.MaxPool2d(kernel_size=pooling_params[0], stride=pooling_params[1]),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 4 * 4, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10)
    )
    model.to(device)
    # torch.nn.init.uniform_(model[0].weight, -init_range, init_range)
    # torch.nn.init.uniform_(model[2].weight, -init_range, init_range)

    # on initialise l'optimiseur
    loss_func = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(model.parameters(), lr=eta)

    for n in range(nb_epochs):
        # on lit toutes les données d'apprentissage
        for x, t in train_loader:
            # on redimensionne x
            x = torch.reshape(x, (batch_size, 1, picture_dim, picture_dim))
            # on calcule la sortie du modèle
            y = model(x)
            # on met à jour les poids
            loss = loss_func(t, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

        # test du modèle (on évalue la progression pendant l'apprentissage)
        acc = 0.
        # on lit toutes les données de test
        for x, t in test_loader:
            # on redimensionne x
            x = torch.reshape(x, (1, 1, picture_dim, picture_dim))
            # on calcule la sortie du modèle
            y = model(x)
            # on regarde si la sortie est correcte
            acc += torch.argmax(y, 1) == torch.argmax(t, 1)
        # on affiche le pourcentage de bonnes réponses
        # print(acc/data_test.shape[0])
        print(str((acc / data_test.shape[0]).item()).replace('.', ','))
    print(f" programme terminé à {datetime.datetime.now().strftime('%H:%M:%S')}")

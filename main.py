from Shallow_network import *

tab_batch_size = list(range(1, 21))
tab_nb_epochs = list(range(1, 21))
tab_learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
tab_hidden_size = [32, 64, 128, 256, 512]
tab_weight_init_range = [(0, 0.1), (-0.1, 0.1), (-0.01, 0.01), (-0.001, 0.001), (-0.0001, 0.0001)]


print("INFLUENCE DE BATCH_SIZE (taille des lots de données pour l'entraînement)")

for batch_size in tab_batch_size :
    
    print("BATCH_SIZE : ", batch_size)
    # Chargement des hyperparamètres
    params = definir_hyperparametres(batch_size=batch_size)

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'])

    # Chargement des données
    model.charger_donnees('mnist.pkl.gz', params)

    # Entraînement du modèle
    model.train_mlp(params)

print("INFLUENCE DE NB_EPOCHS (nombre d'époques d'entraînement)")
for nb_epochs in tab_nb_epochs :
    

    print("NB_EPOCHS : ", nb_epochs)
    # Chargement des hyperparamètres
    params = definir_hyperparametres(nb_epochs=nb_epochs)

    # Initialisation du modèle
    model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'])

    # Chargement des données
    model.charger_donnees('mnist.pkl.gz', params)

    # Entraînement du modèle
    model.train_mlp(params)


print("INFLUENCE DE LEARNING_RATE (taux d'apprentissage pour l'optimiseur)")

for learning_rate in tab_learning_rate :
        
        print("LEARNING_RATE : ", learning_rate)
        
        # Chargement des hyperparamètres
        params = definir_hyperparametres(learning_rate=learning_rate)
    
        # Initialisation du modèle
        model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'])
    
        # Chargement des données
        model.charger_donnees('mnist.pkl.gz', params)
    
        # Entraînement du modèle
        model.train_mlp(params)

print("INFLUENCE DE HIDDEN_SIZE (nombre de neurones dans la couche cachée)")

for hidden_size in tab_hidden_size :
        
        print("HIDDEN_SIZE : ", hidden_size)
        
        # Chargement des hyperparamètres
        params = definir_hyperparametres(hidden_size=hidden_size)
    
        # Initialisation du modèle
        model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'])
    
        # Chargement des données
        model.charger_donnees('mnist.pkl.gz', params)
    
        # Entraînement du modèle
        model.train_mlp(params)

print("INFLUENCE DE WEIGHT_INIT_RANGE (plage d'initialisation des poids)")

for weight_init_range in tab_weight_init_range :
        
        print("WEIGHT_INIT_RANGE : ", weight_init_range)
        
        # Chargement des hyperparamètres
        params = definir_hyperparametres(weight_init_range=weight_init_range)
    
        # Initialisation du modèle
        model = PerceptronMulticouche(params['input_size'], params['hidden_size'], params['output_size'])
    
        # Chargement des données
        model.charger_donnees('mnist.pkl.gz', params)
    
        # Entraînement du modèle
        model.train_mlp(params)


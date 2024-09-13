# Rapport TP GHAZEL et HALVICK

## Partie 1 : Perceptron


1. **`data_train`**
   - **Taille** : `torch.Size([63000, 784])`
   - **Description** : Contient les images d'entraînement, chaque image étant aplatie en un vecteur de 784 pixels (28x28 pixels). La taille totale est `(nombre_d_exemples_d_entraînement, nombre_de_pixels_par_image)`, donc `(63000, 784)`.

2. **`label_train`**
   - **Taille** : `torch.Size([63000, 10])`
   - **Description** : Contient les étiquettes associées aux images d'entraînement, représentées comme des vecteurs one-hot de 10 classes. La taille est `(nombre_d_exemples_d_entraînement, nombre_de_classes)`, donc `(63000, 10)`.

3. **`data_test`**
   - **Taille** : `torch.Size([7000, 784])`
   - **Description** : Contient les images de test, chaque image étant également aplatie en un vecteur de 784 pixels. La taille est `(nombre_d_exemples_de_test, nombre_de_pixels_par_image)`, donc `(7000, 784)`.

4. **`label_test`**
   - **Taille** : `torch.Size([7000, 10])`
   - **Description** : Contient les étiquettes associées aux images de test, représentées comme des vecteurs one-hot de 10 classes. La taille est `(nombre_d_exemples_de_test, nombre_de_classes)`, donc `(7000, 10)`.

5. **`w`**
   - **Taille** : `torch.Size([784, 10])`
   - **Description** : Matrice des poids du modèle. Chaque entrée représente la connexion entre un pixel d'entrée et une classe de sortie. La taille est `(nombre_de_pixels_par_image, nombre_de_classes)`, donc `(784, 10)`.

6. **`b`**
   - **Taille** : `torch.Size([1, 10])`
   - **Description** : Vecteur des biais du modèle, chaque biais étant associé à une classe de sortie. La taille est `(1, nombre_de_classes)`, donc `(1, 10)`.

7. **`y`**
   - **Taille** : `torch.Size([5, 10])`
   - **Description** : Résultat du produit matriciel entre `x` et `w`, plus `b`. La taille est `(batch_size, nombre_de_classes)`. Si `batch_size` est 5, alors `y.shape` sera `(5, 10)` lors de l'entraînement.

8. **`t`**
   - **Taille** : `torch.Size([5, 10])`
   - **Description** : Étiquette vraie pour les exemples courants dans le lot. Représentée comme un vecteur one-hot. La taille est `(batch_size, nombre_de_classes)`. Si `batch_size` est 5, alors `t.shape` sera `(5, 10)`.

9. **`grad`**
   - **Taille** : `torch.Size([5, 10])`
   - **Description** : Gradient de la fonction de perte par rapport aux sorties du modèle (t-y). La taille est `(batch_size, nombre_de_classes)`, donc `(5, 10)` pendant l'entraînement si `batch_size` est 5.

10. **`acc`**
    - **Taille** : `tensor([5970.])`
    - **Description** : Nombre total d'exemples correctement classifiés pendant l'évaluation du modèle. C'est une valeur scalaire accumulée pendant les tests.

11. **`nb_data_test`**
    - **Taille** : `7000`
    - **Description** : Nombre total d'exemples dans l'ensemble de test.

12. **`acc/nb_data_test`**
    - **Taille** : `tensor([0.8533])`
    - **Description** : Précision du modèle sur l'ensemble de test, calculée comme la proportion d'exemples correctement classifiés par rapport au nombre total d'exemples de test. C'est un scalaire qui représente la précision du modèle.

## Partie 2 : Shallow Network
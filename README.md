# K-Means Clustering "From Scratch" - MNIST Dataset

Ce projet est une implémentation manuelle de l'algorithme d'apprentissage non supervisé **K-Means** en Python. Il est appliqué au jeu de données **MNIST** (chiffres manuscrits) pour regrouper les images similaires sans connaître leurs étiquettes au préalable.

L'objectif principal est de comprendre la logique interne de K-Means (calcul de distances, mise à jour des centroïdes, convergence) sans utiliser de librairie de "boîte noire" comme `sklearn.cluster`.

## Fonctionnalités

* **Implémentation pure de K-Means** :
    * Calcul de la distance Euclidienne.
    * Assignation des clusters et recalcule des centroïdes (barycentres).
    * Gestion de la convergence (arrêt si peu de changements).
* **Méthode Elbow (Le Coude)** : Calcul et affichage de l'inertie pour aider à choisir le nombre optimal de clusters ($k$).
* **Visualisation** :
    * Graphique de la méthode Elbow.
    * Affichage des "centroïdes" finaux sous forme d'images (ce à quoi ressemble le "chiffre moyen" de chaque groupe).
* **Analyse des données** : Export des résultats sous forme de DataFrame Pandas montrant quels points appartiennent à quel cluster.

## Prérequis

Pour exécuter ce projet, vous avez besoin de **Python 3.x** et des librairies suivantes :

* `numpy` (Calculs matriciels)
* `matplotlib` (Visualisation)
* `pandas` (Manipulation de données)
* `scikit-learn` (Uniquement pour télécharger le dataset MNIST)

from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Récupère les images des chiffres et les labels correspondant
mnist = fetch_openml("mnist_784") 
mnist.target = mnist.target.astype(np.int8)

x = np.array(mnist.data)
y = np.array(mnist.target)

def distanceEuclidienne(p1, p2): #distance euclidienne car 2 valeures réelles comprises entre 0 et 255
    """Calculer la distance Euclidienne entre deux points p1 et p2."""
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))


def calculate_inertie(x, centroids, cluster_assignement):
    """
    Calculer l'inertie (somme des carrés des distances des points au centroïde de leur cluster)
    
    Args:
    x (np.array): Ensemble des points
    centroids (list): Liste des centroïdes
    cluster_assignement (list): Attribution des points aux clusters
    
    Returns:
    float: Valeur de l'inertie
    """
    inertie = 0
    for i, point in enumerate(x):
        cluster = cluster_assignement[i]
        inertie += distanceEuclidienne(point, centroids[cluster]) ** 2
    return inertie

def elbow_method(x, max_k=10):
    """
    Méthode Elbow pour déterminer le nombre optimal de clusters
    
    Args:
    x (np.array): Ensemble des points
    max_k (int): Nombre maximum de clusters à tester
    
    Returns:
    list: Liste des inerties pour différentes valeurs de k
    """
    inerties = []
    
    for k in range(1, max_k + 1):
        print(f"Calcul pour k = {k}")
        centroids, cluster_assignement = kmeans(x, k, max_iter=30)
        inertie = calculate_inertie(x, centroids, cluster_assignement)
        inerties.append(inertie)
    
    return inerties


def kmeans(x, k, max_iter=100):
    # Convertir x en liste de listes pour faciliter la manipulation
    x_list = [list(point) for point in x]
    
    # Initialiser les centroïdes aléatoirement parmi les points
    centroids = random.sample(x_list, k)
    
    # Liste pour stocker les labels des clusters
    cluster_assignement = [-1] * len(x_list)
    
    for iteration in range(max_iter):
        # Étape 1 : Affecter chaque point au centroïde le plus proche
        new_cluster_assignement = []
        
        for i in range(len(x_list)): #pour chaque point
            min_distance = float('inf')
            label = None
            for j in range(k): #compare au centroïdes
                dist = distanceEuclidienne(x_list[i], centroids[j])
                if dist < min_distance:
                    min_distance = dist
                    label = j
            new_cluster_assignement.append(label) #new label c'est un tableau de taille len(x) où indice (i) = point et valeur (tab[i]) = centroide associé
        
        # Étape 2 : Recalculer les nouveaux centroïdes
        new_centroids = []
        for i in range(k):
            cluster_points = [x_list[j] for j in range(len(x_list)) if new_cluster_assignement[j] == i] #crée un tableau de tout les points appartenant au cluster k
            # Calculer la moyenne des points dans le cluster pour obtenir un nouveau centroïde
            if cluster_points:  # Eviter la division par zéro si un cluster est vide
                mean_point = [sum(coord) / len(cluster_points) for coord in zip(*cluster_points)]
                new_centroids.append(mean_point)
            else:
                # Si un cluster est vide, on choisit un centroïde aléatoire parmi les points
                new_centroids.append(random.choice(x_list))
        
        # Vérification de la   convergence
        changed = sum(1 for a, b in zip(cluster_assignement, new_cluster_assignement) if a != b)
        print(f"Itération {iteration + 1}: {changed} points ont changé de cluster")
        
        if changed == 0 or iteration == max_iter - 1 or changed < 5:
            if changed == 0:
                print(f"Convergence attein3021te après {iteration + 1} itérations.")
            elif changed < 5:
                print(f"il y a moins de ({changed}) points qui ont changés de cluster")
                print(f"Nombre maximum d'itérations ({max_iter}) atteint sans convergence.")
            break
            
        cluster_assignement = new_cluster_assignement
        centroids = new_centroids
    
    return centroids, cluster_assignement

#EXECTUTION DE LA METHODE ELBOW
n_samples = 2000
x_subset = x[:n_samples]

inerties = elbow_method(x_subset)

# Visualisation de la courbe Elbow
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(inerties) + 1), inerties, marker='o')
plt.title('Méthode Elbow pour déterminer le nombre optimal de clusters')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Inertie')
plt.xticks(range(1, len(inerties) + 1))
plt.grid(True)
plt.show()

# Nombre de clusters
while True:
    try:
        k = int(input("\nChoisissez le nombre de clusters (k) que vous voulez utiliser : "))
        if k > 0 and k <= 10:
            break
        else:
            print("Veuillez choisir un k entre 1 et 10.")
    except ValueError:
        print("Veuillez entrer un nombre valide.")

centroids, cluster_assignement = kmeans(x_subset, k, max_iter=30)  # Limitation à 30 itérations pour éviter les boucles infinies
#Toutes les donées des points sont stockés dans cluster asignement, lindice i = le point, la valeur tab[i] = le cluster assigné


def trier_donnees(tab):
    """trie le tableau de clusters en un dictionnaire avec keys -> k | values -> point assignés a k"""
    tableau_range = {}

    for centroides in range(k):
        tableau_range[centroides] = []

    for i in range(len(cluster_assignement)):
        tableau_range[cluster_assignement[i]].append(i)

    return tableau_range



# Exécution de K-Means sur un sous-ensemble de MNIST


  
# Affichage des centroïdes obtenus
fig, axes = plt.subplots(2, 5, figsize=(10, 8))
axes = axes.ravel()
for i in range(k):
    axes[i].imshow(np.array(centroids[i]).reshape(28, 28), cmap="gray")
    axes[i].axis("off")
plt.suptitle("Centroïdes après K-Means (représentation des clusters)")
plt.tight_layout()
plt.show()

#affichage des données triées
df = pd.DataFrame.from_dict(trier_donnees(cluster_assignement), orient='index')

print(df)
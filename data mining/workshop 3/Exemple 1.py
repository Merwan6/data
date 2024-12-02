#Workshop 3 

#-------------------------------------------------------Importation des bibliothèques----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx
import warnings

#Configure l'esthétique des graphique généré par seaborn
sns.set(style="darkgrid", color_codes=True)

#Configure toutes les sorties avec un affichage à 75 colonnes
pd.set_option('display.max_columns', 75)

#--------------------------------------------------------Dataset--------------------------------------------------------------------------------
"""Retourne le nombre d'éléments par colonne ie colonne 0 ==> 7501 éléments,..."""
data = pd.read_csv(".\\Market_Basket_Optimisation.csv", header=None)  #Ouverture du fichier chemin relatif + header = None pour mettre mettre en nom de colonnes des nombres

#Affichage des infos de tout le tableau
data.info()    

#Affichage des infos des 5 premières lignes (de 0 à 4)
data.head() 

#Renvoie : 
"""Pour les colonnes numériques :
count : Nombre de valeurs non nulles.
mean : Moyenne des valeurs.
std : Écart type (indicateur de dispersion).
min : Valeur minimale.
25% : Premier quartile (25% des données en dessous de cette valeur).
50% : Médiane ou deuxième quartile (valeur centrale, 50%).
75% : Troisième quartile (75% des données en dessous de cette valeur).
max : Valeur maximale."""
data.describe()

#----------------------------------------------------------Analyse Exploratoire des données----------------------------------------------------
#Affichage de la 1ère colonne en graphique
"""plt.cm.rainbow : Utilise le colormap "rainbow" de Matplotlib.
np.linspace(0, 1, 40) : Génère 40 valeurs uniformément réparties entre 0 et 1. Ces valeurs sont utilisées pour sélectionner les couleurs dans le dégradé.
La variable color contient une liste de 40 couleurs."""
color = plt.cm.rainbow(np.linspace(0, 1, 40))

"""data[0] : Accède à la première colonne du DataFrame data (peut être une erreur, il est préférable d'utiliser data.iloc[:, 0] si les colonnes n'ont pas de nom).
.value_counts() : Compte combien de fois chaque valeur apparaît dans la colonne.
.head(40) : Prend les 40 premières valeurs les plus fréquentes.
.plot.bar() : Crée un graphique en barres.
color=color : Attribue les 40 couleurs générées précédemment.
figsize=(13,5) : Définit la taille du graphique (largeur : 13, hauteur : 5)."""
data[0].value_counts().head(40).plot.bar(color = color, figsize=(13,5))

"""plt.title : Ajoute un titre au graphique.
fontsize=20 : Définit la taille de la police du titre."""
plt.title('frequency of most popular items', fontsize = 20)

"""plt.xticks(rotation=90)"""
plt.xticks(rotation = 90 )

"""Enlève une grille"""
plt.grid()

"""Montre le graphique"""
plt.show()

#Ajoute une nouvelle colonne au DataFrame data
data['food'] = 'Food'

#Tronque les lignes avant 0 et après 15
food = data.truncate(before = -1, after = 15)

#Transforme le dataFrame en graphe
"""food :
C'est le DataFrame que vous transformez en graphe.
Chaque ligne du DataFrame représente une arête (ou connexion) entre deux nœuds.

source :
Spécifie la colonne du DataFrame qui contient les nœuds source des arêtes.
Dans votre cas, c'est la colonne 'food'.

target :
Spécifie la colonne du DataFrame qui contient les nœuds cible des arêtes.
Vous utilisez ici 0, qui est probablement la première colonne du DataFrame.

edge_attr=True :
Si activé, toutes les colonnes restantes du DataFrame sont ajoutées comme attributs pour les arêtes.
Si vous voulez inclure uniquement certaines colonnes comme attributs d’arêtes, vous pouvez passer une liste de noms de colonnes, par exemple : edge_attr=['weight', 'type']."""
food = nx.from_pandas_edgelist(food, source = 'food', target = 0, edge_attr = True)

#Module warnings utilisé pour signaler de potentiels problèmes ==> message d'alerte
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (13, 13)                                           #Taille de la figure
pos = nx.spring_layout(food)                                                        #Génère des noueds de avec l'algorithme de force Nx
color = plt.cm.Set1(np.linspace(0, 15, 1))                                          #Génère des couleurs depuis le set1
nx.draw_networkx_nodes(food, pos, node_size = 15000, node_color = color)            #Trace les noeuds sur le graphe
nx.draw_networkx_edges(food, pos, width = 3, alpha = 0.6, edge_color = 'black')     #Trace les arêtes des graphes
nx.draw_networkx_labels(food, pos, font_size = 20, font_family = 'sans-serif')      #Ajoute les étiquettes aux noeuds
plt.axis('off')                                                                     #Enlève les axes
plt.grid()                                                                          #N'affiche pas la grille
plt.title('Top 15 First Choices', fontsize = 20)                                    #Affiche le titre
plt.show()                                                                          #Affiche le graphe



#Obtenir une liste de transactions 
transactions = []
for i in range(0, min(10, len(data))) :
    transactions.append([str(data.values[i,j]) for j in range(0, len(data.columns))])

transactions[:1]
   
#--------------------------------------------------Association rules------------------------------------------------------------------------



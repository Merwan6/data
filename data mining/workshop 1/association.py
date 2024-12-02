#Premier algorithe d'IA ==> Assocation

#Importation des bibliothèques
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

#Importation des jeux de données
store_data = pd.read_csv(".\\store_data.csv",header=None)   #header = None décale la 1ère ligne 
store_data.head()

#Traitement des données

#Convertir le dataframe pandas en une liste de listes
records = []
for i in range(0,7501):
    records.append([str(store_data.values[i,j]) for j in range(0,20)])

#Application d'Apriori
association_rules = apriori(records, min_support = 0.0045, min_confidence = 0.2, min_lift = 3 , min_lenght = 2)
association_results = list(association_rules)

#Affichage des résultats
print(len(association_results))
print(association_results[0])

#Affichage plus claire

for item in association_results: 
    
    #premier indice pour la liste intérieure
    #Contient les items de base et les items ajoutés
    
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + "->" + items[1])
    
    #Deuxième indice de la liste intérieure
    print("Support: " + str(item[1]))
    
    #Troisième indice de la liste localisée at la position 0
    #Troisième index de la liste intérieure
    
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("======================================")
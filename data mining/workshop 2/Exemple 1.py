#Exemple 1

#Importation des bibliothèques
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Données

data = {"ID" : [1,2,3,4,5,6],
        "Onion" : [1,0,0,1,1,1],
        "Potato" : [1,1,0,1,1,1],
        "Burger" : [1,1,0,0,1,1],
        "Milk" : [0,1,1,1,0,1],
        "Beer" : [0,0,1,0,1,0]}

#Transforme en tableau
df = pd.DataFrame(data) 
print(df)

df = df[["Onion", "Potato", "Burger", "Milk", "Beer"]]
print(df)


#Génération des itemsets basé sur les supports
frequent_itemsets = apriori(df,min_support = 0.5, use_colnames=True)
print(frequent_itemsets)
taille = frequent_itemsets.shape[0]

#Génération des règles correspondant au support, confidence, lift (et leverage et conviction)
rules = association_rules(frequent_itemsets ,taille, metric="lift" , min_threshold = 1)
print(rules)

#Interprétation des résultats
rules[(rules["lift"]>1.125) & (rules["confidence"]> 0.8)]


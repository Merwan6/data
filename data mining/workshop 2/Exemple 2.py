#Exemple 2

#Importation des bibliothèques
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Données
retail_shopping_basket = {"ID" : [1,2,3,4,5,6],
                          'Basket':[['Beer', 'Diaper', 'Pretzels', 'Chips', 'Aspirin'],
                                  ['Diaper', 'Beer', 'Chips', 'Lotion', 'Juice', 'BabyFood', 'Milk'],
                                  ['Soda', 'Chips', 'Milk'],
                                  ['Soup', 'Beer', 'Diaper', 'Milk', 'IceCream'],
                                  ['Soda', 'Coffee', 'Milk', 'Bread'],
                                  ['Beer', 'Chips']
                                 ]
                        }

#Transformer le dico en tableau
retail = pd.DataFrame(retail_shopping_basket)
print(retail)
retail = retail[["ID","Basket"]]

# Changer la largeur maximale de colonne
pd.options.display.max_colwidth = 100
print(retail)

#Encodage du basket
retail = retail.drop("Basket",axis=1).join(retail.Basket.str.join(",").str.get_dummies(","))   #drop ==> enlève
print(retail)

#Apriori
frequent_itemsets_2 = apriori(retail.drop("ID",axis = 1), use_colnames=True)   #valeur par défaut min_supp = 0.5
print(frequent_itemsets_2)

#Règles d'associations
rules = association_rules(frequent_itemsets_2,1, metric = "confidence" )
print(rules)

print(association_rules(frequent_itemsets_2,1)) #prends par défaut la valeur de metric = "confidence"
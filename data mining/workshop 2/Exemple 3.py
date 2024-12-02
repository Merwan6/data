#Exemple 3

#Importation des bibliothèques
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt


#Ouverture du fichier 
movies = pd.read_csv(".\\movies.csv")
movies.head(10)

#Tranformation des données
movies_ohe = movies.drop('genres',axis = 1).join(movies.genres.str.get_dummies())

#Largeur des colonnes
pd.options.display.max_columns = 100

#On enlève la 1ère colonne de movies_ohe et compte la fréquence de chaque valeur de chaque colonne 
stat1 = movies_ohe.drop(["title", "movieId"],axis = 1).apply(pd.value_counts)
print(stat1)

#On transpose en ayant enlevé la colonne 0, en triant la colonne 1 et en renommant la colonne 1
stat1 = stat1.transpose().drop(0,axis = 1).sort_values(by=1, ascending=False).rename(columns={1:'No. of movies'})
print(stat1)

#Colonne genres est transformé en une liste de genre pour chaque film, séparé par |, compte le nombre d'éléments, reset l'index pour créer un nouveau dataframe, rename les colonnes
stat2 = movies.join(movies.genres.str.split('|').reset_index().genres.str.len(), rsuffix='r').rename(columns={'genresr':'genre_count'})
print(stat2)

#Filtre les lignes ou la colonne "genre_count" = 1, enlève la colonne movieId, groupe les données par la colonne "genres", somme les colonnes restantes, trie les résultats selon la colonne "genre_count")
stat2 = stat2[stat2['genre_count']==1].drop(["title", "movieId"],axis = 1).groupby('genres').sum().sort_values(by='genre_count', ascending=False)
print(stat2)

#Fusion entre deux dataframe, how="left" ==> toutes les lignes de stat1 seront conservées, indice des deux dataframes sont utilisées comme clé de jointure, fillna(0) ==> remplace toutes les valeurs NaN par 0
stat = stat1.merge(stat2, how='left', left_index=True, right_index=True).fillna(0)
print(stat)

#Conversion de la colonne genre_count en entier
stat.genre_count=stat.genre_count.astype(int)

#Renommage de la colonne genre_count en No.of movies...
stat.rename(columns={'genre_count': 'No. of movies with only 1 genre'},inplace=True)
print(stat)

#Affiche directement dans le notebook
%matplotlib inline

#Tracer le graphique
movies_ohe.set_index(['movieId','title']).sum(axis=1).hist()

#Titre du graphique
plt.title('distribution of number of genres')

#Analyse 
movies_ohe.set_index(['movieId','title'],inplace=True)

#Algorithme apriori ==> classe les minimum support
frequent_itemsets_movies = apriori(movies_ohe,use_colnames=True, min_support=0.025)
print(frequent_itemsets_movies)

#Association de règles par rapport à lift, seuil supérieur à 1.25
rules_movies =  association_rules(frequent_itemsets_movies,1, metric='lift', min_threshold=1.25)
print(rules_movies)

#Conviction > 1.25
print(rules_movies[(rules_movies.conviction>1.25)])

#Nombre de lignes à afficher ==> 50
pd.options.display.max_rows=50

#Conviction > 1.25
print(rules_movies[(rules_movies.conviction>1.25)])

#Vérifier si la colonnes "genres" contient le mot "Adventures", "genres" ==> "Children", "genres" ==> "Animation"
A = movies[(movies.genres.str.contains('Adventure')) & (movies.genres.str.contains('Children')) & (~movies.genres.str.contains('Animation'))]
print(A)

#Nombre de lignes à afficher ==> 50
pd.options.display.max_rows=500

#Vérifier si la colonnes "genres" contient le mot "Adventures", "genres" ==> "Children", "genres" ==> "Animation"
A = movies[(movies.genres.str.contains('Adventure')) & (movies.genres.str.contains('Children')) & (~movies.genres.str.contains('Animation'))]
print(A)

""" To recap, a straightforward 4-steps approach to association rule:

One-hot encone the basket in dataframe.
Generate frequent itemsets using apriori.
Generate rule with association_rules.
Interpret & evalute the result with metrics."""
#FP-GROWTH

#------------------------------------------------------Bibliothèques---------------------------------------------------
from fpgrowth_py.fpgrowth import *
from fpgrowth_py.utils import *
from collections import defaultdict, OrderedDict
from csv import reader
from itertools import chain, combinations
from optparse import OptionParser
from fpgrowth_py.utils import *

#Fonction fpgrowth
def fpgrowthFromFile(fname, minSupRatio, minConf): 
    """Cette fonction compte le nombre d'item présent dans chaque transaction et retourne ceux qui dont la fréquence passe le minsup
    
        fname : nom du fichier contenant les données des transactions
        minSupRatio : Le seuil de support minimal, exprimé comme un ratio par rapport au nombre total de transactions.
        minConf : Le seuil de confiance minimale pour générer des règles d'association."""
    
    itemSetList, frequency = getFromFile(fname)                             #Chargement du fichier
    minSup = len(itemSetList) * minSupRatio                                 #Le support minimal est calculé en multipliant le ratio (minSupRatio) par le nombre total de transactions.
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)     #La fonction constructTree construit l'arbre FP (Frequent Pattern Tree) ainsi qu'une table de tête (headerTable), qui est utilisée pour suivre les occurrences des éléments fréquents.

    freqItems = []                                                          #Initialisation d'une liste à 0
    mineTree(headerTable, minSup, set(), freqItems)                         #La fonction mineTree explore l'arbre FP pour extraire les itemsets fréquents, en fonction du seuil de support minimum.
    rules = associationRule(freqItems, itemSetList, minConf)                #Une fois les itemsets fréquents extraits, la fonction associationRule génère des règles d'association ayant une confiance supérieure ou égale au seuil minConf.
    return freqItems, rules                                                 #freqItems : La liste des itemsets fréquents extraits de l'arbre FP.
                                                                            #rules : Les règles d'association générées à partir des itemsets fréquents.

#Construction de l'arbre
def constructTree(itemSetList, frequency, minSup):
    """Cette fonction construit l'arbre FP à partir d'un ensemble de transactions (itemSetList) et de leurs fréquences associées (frequency). 
       Elle filtre également les éléments ayant un support inférieur au seuil minimum (minSup)."""
    
    #Table d'en-tête initialisé pour contenir la fréquence des éléments dans toutes les transactions
    headerTable = defaultdict(int)         
                                                  
    #Chaque élément de chaque transaction est ajouté à la table d'en-tête avec son support cumulé
    for idx, itemSet in enumerate(itemSetList):
        for item in itemSet:
            headerTable[item] += frequency[idx]

    #Enlève les élements dont la fréquence est inférieure au seil minimum (minSup)
    headerTable = dict((item, sup) for item, sup in headerTable.items() if sup >= minSup)
    if(len(headerTable) == 0):
        return None, None                       #Si aucun éléments n'est retourné ==> retourne None

    #Initalisation de la table d'en-tête
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]

    #Création de la racine de l'arbre FP
    fpTree = Node('Null', 1, None)
    
    #Mise à jour de l'arbre pour chaque itemset nettoyé
    for idx, itemSet in enumerate(itemSetList):
        itemSet = [item for item in itemSet if item in headerTable]
        itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
        currentNode = fpTree
        for item in itemSet:
            currentNode = updateTree(item, currentNode, headerTable, frequency[idx])

    return fpTree, headerTable

#Mise à jour de l'arbre
def updateTree(item, treeNode, headerTable, frequency):
    """Cette fonction insère un élément dans l'arbre FP en partant du nœud courant (treeNode)."""
    
    if item in treeNode.children:
        
        #Si l'item existe, incrémente le compteur
        treeNode.children[item].increment(frequency)
    else:
        #Sinon, crée une nouvelle branche
        newItemNode = Node(item, frequency, treeNode)
        treeNode.children[item] = newItemNode
        #Relie la branche à la table d'en-tête
        updateHeaderTable(item, newItemNode, headerTable)

    return treeNode.children[item]

#Mise à jour de la table d'en-tête
def updateHeaderTable(item, targetNode, headerTable):
    """Cette fonction met à jour la structure de la table d'en-tête pour lier un nouvel élément (targetNode)
        à la liste des nœuds associés à un élément donné (item)."""
    
    #Création d'un lien initial
    if(headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        #Le noeud existe déjà
        currentNode = headerTable[item][1]
        #La fonction parcourt les noeuds chaînés jusqu'au dernier et y ajoute le targetNode
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode

#Exploration de l'arbre
def mineTree(headerTable, minSup, preFix, freqItemList):
    """Cette fonction explore récursivement un arbre FP pour extraire les itemsets fréquents. 
        Elle applique l'idée centrale de croissance de motifs (pattern growth), 
        où de nouveaux motifs fréquents sont construits à partir des motifs précédents en explorant des sous-arbres conditionnels.
        
        headerTable : La table d'en-tête de l'arbre FP, contenant des informations sur les éléments fréquents et leurs liens 
                        vers les nœuds correspondants dans l'arbre FP.
        minSup : Le support minimum requis pour qu'un motif soit considéré comme fréquent.
        preFix : Un ensemble d'éléments représentant le préfixe actuel (motif déjà construit jusqu'ici).
        freqItemList : Une liste pour stocker les itemsets fréquents extraits.
    """
    
    #Trie les éléments de la table par la fréquence et les ajoute à une liste
    sortedItemList = [item[0] for item in sorted(list(headerTable.items()), key=lambda p:p[1][0])] 
   
    #Parcourt les éléments triés en commençant par la fréquence la plus basse
    for item in sortedItemList:  
        #Pour chaque éléments dans la liste triée, un nouveau itemset est crée
        newFreqSet = preFix.copy()
        newFreqSet.add(item)
        freqItemList.append(newFreqSet)
        
        #Trouve toutes les combinaisons possibles avec leurs fréquences
        conditionalPattBase, frequency = findPrefixPath(item, headerTable) 
        
        #Construction du sous-arbre conditionnel
        conditionalTree, newHeaderTable = constructTree(conditionalPattBase, frequency, minSup) 
        
        #Exploration du sous-arbre
        if newHeaderTable != None:
            #Si le sous-arbre n'existe pas, il faut le crée.
            mineTree(newHeaderTable, minSup,
                       newFreqSet, freqItemList)

#
def findPrefixPath(basePat, headerTable):
    """Cette fonction identifie les chemins conditionnels (prefix paths) pour un élément donné dans un arbre FP. 
       Les chemins conditionnels sont des séquences d'éléments qui mènent à un nœud particulier dans l'arbre, en excluant le nœud lui-même.
       
       basePat : L'élément pour lequel on veut trouver les chemins conditionnels.
       headerTable : La table d'en-tête contenant des informations sur les éléments fréquents, 
                   y compris des liens vers leurs nœuds correspondants dans l'arbre FP."""
       
    #Le premier noued est associé à l'élément (basePat) ie deux listes sont crées pour stocker les chemins conditionnels / les fréquences correspondantes
    treeNode = headerTable[basePat][1] 
    condPats = []
    frequency = []
    
    #Parcours des noueds associés
    while treeNode != None:
        prefixPath = []
        
        #Extraction du chemin jusqu'à la racine
        ascendFPtree(treeNode, prefixPath)  
        
        #Filtrage des éléments
        if len(prefixPath) > 1:
            
            #Le chemin est tronqué pour exclure l'élément de base (basePat)
            condPats.append(prefixPath[1:])
            frequency.append(treeNode.count)

        #Passage au noued suivant
        treeNode = treeNode.next  
    return condPats, frequency

#
def ascendFPtree(node, prefixPath):
    """La fonction ascendFPtree remonte depuis un nœud donné jusqu'à la racine de l'arbre FP. 
       Elle collecte les noms des éléments rencontrés sur le chemin dans une liste prefixPath."""
       
    if node.parent != None:
        prefixPath.append(node.itemName)
        ascendFPtree(node.parent, prefixPath)
Projet Spark MLlib · Chicago Crime · 1 000 000 crimes

# Comment le projet fonctionne

Pourquoi Spark, ce que deviennent les données, comment elles sont séparées, et ce que les modèles ont appris. Tous les chiffres viennent du run complet sur 1M de lignes.

Question 1

## Pourquoi Spark ?

Le dataset complet de Chicago fait plus de 8 millions de lignes (\~2 Go). Pandas charge tout en mémoire sur une seule machine et sur un seul cœur. Spark découpe les données en **partitions** et les traite en parallèle, sur tous les cœurs du PC ou sur un cluster de plusieurs machines, **avec le même code**.

| Critère | Pandas / scikit-learn | Hadoop MapReduce | Apache Spark |
| --- | --- | --- | --- |
| Exécution | 1 machine, 1 cœur | Distribuée | Distribuée |
| Données intermédiaires | RAM | Écrites sur disque à chaque étape | Gardées en mémoire (cache) |
| Limite de taille | La RAM de la machine | Aucune pratique | Aucune pratique |
| Machine learning | scikit-learn (non distribué) | Mahout (abandonné) | MLlib intégré, distribué |
| Itératif (ML) | Rapide si ça tient en RAM | Très lent (relit le disque) | Rapide (jusqu'à \~100× vs MapReduce) |
| Langages | Python | Java | Python, Scala, Java, R, SQL |

Les algorithmes de ML sont **itératifs** : la régression logistique repasse 50 fois sur les données, KMeans aussi. Avec MapReduce chaque passage relit le disque ; Spark garde les données en mémoire avec `df.cache()`. C'est la raison principale du choix.

### Ce qui s'est passé sur ton PC

Mode `local[*]` : le **driver** (ton script Python) découpe le travail en tâches, et 16 **executors** légers (un par cœur logique) les exécutent en parallèle. Dans les logs tu voyais `(14 + 2) / 16` : 14 tâches finies, 2 en cours, sur 16.

Question 2

## Comment les données ont été séparées

Trois étapes : nettoyage, puis découpage aléatoire 80 / 20 avec `randomSplit([0.8, 0.2], seed=42)`. Le `seed` fixe le hasard : relancer donne exactement le même découpage, donc des résultats reproductibles.

### Nettoyage : 8 443 lignes retirées

- date illisible
- latitude / longitude manquantes
- coordonnées hors de Chicago (lat ≤ 41 ou long ≥ −87)
- type de crime ou district manquant

### Pourquoi 80 / 20 ?

Le modèle apprend sur le **train**. Le **test** est caché pendant l'entraînement : on mesure la qualité sur des crimes que le modèle n'a jamais vus. Sinon il pourrait juste « mémoriser ».

### Le déséquilibre des classes

Seulement **14,18 %** des crimes mènent à une arrestation. Un modèle qui répond toujours « pas d'arrestation » aurait déjà 86 % d'accuracy sans rien apprendre. On donne donc un **poids** à chaque ligne pour que les deux classes comptent autant :

Question 3

## Le pipeline MLlib

MLlib ne comprend que des nombres réunis dans un seul vecteur `features`. Le `Pipeline` enchaîne les transformations puis le modèle, et se sauvegarde en un seul bloc dans `models/`.

**StringIndexer**"THEFT" → 0\
"BATTERY" → 1

→

**OneHotEncoder**1 → \[0,1,0,0…\]\
pas d'ordre artificiel

→

**VectorAssembler**catégories + heure, jour, district, lat, long → `features`

→

**Modèle**LogisticRegression ou RandomForest → 0 / 1 + probabilité

Transformer = applique une règle (`transform`). Estimator = apprend sur les données (`fit`) et produit un Transformer. Un Pipeline entier est un Estimator.

Question 4

## Résultats

| Modèle | AUC | Accuracy | F1 | Entraînement |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0,835 | 73,4 % | 0,771 | 2 min 13 s |
| Random Forest (50 arbres) | 0,835 | 76,3 % | 0,793 | 4 min 11 s |

**AUC** : probabilité que le modèle classe une vraie arrestation plus haut qu'une non-arrestation (0,5 = hasard, 1 = parfait). **F1** : équilibre entre précision et rappel.

### Matrice de confusion — Random Forest, sur les 198 403 crimes du test

Prédit : non

Prédit : oui

Réel : non

**131 012**correct

**39 018**fausse alerte

Réel : oui

**7 930**arrestation ratée

**20 443**correct

Le modèle retrouve **72 %** des vraies arrestations (20 443 / 28 373). En échange il donne beaucoup de fausses alertes : c'est l'effet voulu des poids, qui favorisent la classe rare.

### Clustering KMeans : zones chaudes

KMeans regroupe les crimes par position (latitude, longitude) sans étiquette. On a testé k = 4, 6, 8, 10 et gardé celui avec le meilleur **score silhouette** (proche de 1 = groupes bien séparés) : **k = 10, silhouette 0,53**. La zone la plus « arrêtée » a 20,8 % d'arrestations, la moins arrêtée 9,2 %.

Question 5

## Voir la Spark UI

### En direct · port 4040

Existe seulement tant qu'une session Spark tourne. Pas besoin d'entraîner : un shell suffit.

```
. .\env.ps1
pyspark
>>> df = spark.read.parquet("data/crimes.parquet")
>>> df.groupBy("primary_type").count().show()
```

Puis ouvrir `http://localhost:4040`. Laisser le shell ouvert.

### Anciens runs · port 18080

Le History Server relit les logs de chaque `spark-submit` passé, même après la fin du programme.

```
.\history.ps1
```

Puis ouvrir `http://localhost:18080`. Ctrl+C pour arrêter.

Onglets à montrer pendant la démo : **Jobs** (chaque action Spark), **Stages** (les 16 tâches parallèles), **Storage** (les DataFrames en cache), **SQL** (le plan d'exécution), **Executors** (mémoire et CPU).
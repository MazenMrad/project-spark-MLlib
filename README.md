# Spark MLlib — Chicago Crime

Présentation, installation, configuration et manipulation d'Apache Spark avec MLlib,
sur le dataset [Chicago Crimes 2001-present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2).

## Contenu
- `chicago_crime_mllib.ipynb` — prototype complet (chargement, nettoyage, exploration, LR / Random Forest, KMeans, sauvegarde)
- `data/` — mettre le CSV ici (`data/Crimes.csv`), ignoré par git
- `models/` — modèles sauvegardés, ignorés par git

## Installation (Windows)
1. **Java 17** (Spark 3.5 ne supporte pas Java 25) → `JAVA_HOME`
2. **Python 3.11** (PySpark 3.5 ne supporte pas encore Python 3.13 de façon fiable)
3. `winutils.exe` + `hadoop.dll` (Hadoop 3.3) dans `C:\hadoop\bin` → `HADOOP_HOME=C:\hadoop`
4. ```
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   jupyter notebook
   ```

## Utilisation
- Données complètes : dans la cellule 2, `SOURCE = "csv"` et `CSV_PATH` vers ton fichier
- Démo légère sans téléchargement : `SOURCE = "api"` (API Socrata)
- Pendant le développement : `SAMPLE = 0.1`
- Spark UI : http://localhost:4040

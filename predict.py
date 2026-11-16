"""Charge le modèle Random Forest sauvegardé et prédit l'arrestation pour quelques crimes.
Usage : python predict.py
"""
import os, sys, glob

# Utilise le Java 17 + winutils portables du projet (Spark 3.5 ne marche pas avec Java 25)
ROOT = os.path.dirname(os.path.abspath(__file__))
jdk = glob.glob(os.path.join(ROOT, ".tools", "jdk-17*"))
if jdk:
    os.environ["JAVA_HOME"] = jdk[0]
    os.environ["HADOOP_HOME"] = os.path.join(ROOT, ".tools", "hadoop")
    os.environ["PATH"] = os.pathsep.join([os.path.join(jdk[0], "bin"),
                                          os.path.join(ROOT, ".tools", "hadoop", "bin"),
                                          os.environ["PATH"]])
os.environ["PYSPARK_PYTHON"] = sys.executable
os.chdir(ROOT)

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = (SparkSession.builder.appName("Predict").master("local[*]")
         .config("spark.driver.memory", "2g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

model = PipelineModel.load("models/rf_arrest")

cols = ["primary_type", "location_description", "hour", "day_of_week", "month",
        "is_weekend", "district", "domestic", "latitude", "longitude"]
crimes = [
    ("NARCOTICS",         "STREET",      22, 6, 7, 1, 11, 0, 41.88, -87.72),
    ("THEFT",             "APARTMENT",   14, 3, 3, 0,  8, 0, 41.77, -87.69),
    ("WEAPONS VIOLATION", "STREET",       1, 7, 8, 1,  7, 0, 41.78, -87.66),
    ("BATTERY",           "RESIDENCE",    2, 1, 1, 1,  6, 1, 41.75, -87.63),
    ("MOTOR VEHICLE THEFT", "STREET",    20, 4, 11, 0, 18, 0, 41.90, -87.63),
]

pred = model.transform(spark.createDataFrame(crimes, cols))
for r in pred.select("primary_type", "location_description", "hour", "prediction", "probability").collect():
    p = r["probability"][1] * 100
    verdict = "ARRESTATION" if r["prediction"] == 1 else "pas d'arrestation"
    print(f"{r['primary_type']:<22} {r['location_description']:<12} {r['hour']:>2}h  -> {verdict:<18} ({p:.1f}%)")

spark.stop()

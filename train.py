"""Job Spark complet : chargement -> nettoyage -> exploration -> MLlib -> résultats.

Lancement :
    . .\\env.ps1
    spark-submit --driver-memory 8g train.py            # 1M lignes (~10 min)
    spark-submit --driver-memory 8g train.py 0.1        # échantillon 10 % (~1 min)

Tout est écrit dans results/ et models/.
"""
import os, sys, time, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator, ClusteringEvaluator)

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
SAMPLE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
CSV, PARQUET, OUT = "data/Crimes.csv", "data/crimes.parquet", "results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

spark = (SparkSession.builder.appName("ChicagoCrime-MLlib")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
log(f"Spark {spark.version} — UI : {spark.sparkContext.uiWebUrl}")

# ---------- 1. Chargement ----------
if not os.path.exists(PARQUET):
    log("Conversion CSV -> Parquet...")
    raw = spark.read.csv(CSV, header=True)
    for c in raw.columns:
        raw = raw.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
    raw.write.mode("overwrite").parquet(PARQUET)
raw = spark.read.parquet(PARQUET)

# ---------- 2. Nettoyage ----------
to_bool = lambda c: F.when(F.lower(F.col(c)) == "true", 1).otherwise(0)
df = (raw
      .withColumn("ts", F.coalesce(F.try_to_timestamp(F.col("date"), F.lit("MM/dd/yyyy hh:mm:ss a")),
                                   F.try_to_timestamp(F.col("date"))))
      .withColumn("arrest", to_bool("arrest")).withColumn("domestic", to_bool("domestic"))
      .withColumn("latitude", F.col("latitude").cast("double"))
      .withColumn("longitude", F.col("longitude").cast("double"))
      .withColumn("district", F.col("district").cast("double").cast("int"))
      .dropna(subset=["ts", "latitude", "longitude", "primary_type", "district"])
      .filter((F.col("latitude") > 41) & (F.col("longitude") < -87))
      .withColumn("hour", F.hour("ts")).withColumn("day_of_week", F.dayofweek("ts"))
      .withColumn("month", F.month("ts")).withColumn("year", F.year("ts"))
      .withColumn("is_weekend", F.col("day_of_week").isin(1, 7).cast("int"))
      .fillna({"location_description": "UNKNOWN"}))
if SAMPLE < 1:
    df = df.sample(fraction=SAMPLE, seed=42)
df = df.cache()
n = df.count()
log(f"{n:,} lignes après nettoyage (sample={SAMPLE})")

# ---------- 3. Exploration ----------
top = (df.groupBy("primary_type").agg(F.count("*").alias("n"),
                                      F.round(F.avg("arrest") * 100, 1).alias("arrest_pct"))
         .orderBy(F.desc("n")).toPandas())
top.to_csv(f"{OUT}/crimes_par_type.csv", index=False)
hours = df.groupBy("hour").count().orderBy("hour").toPandas()
dist = df.groupBy("district").agg(F.avg("arrest").alias("r")).orderBy("district").toPandas()

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
t10 = top.head(10)
ax[0].barh(t10["primary_type"][::-1], t10["n"][::-1]); ax[0].set_title("Top 10 types de crimes")
ax[1].plot(hours["hour"], hours["count"], marker="o"); ax[1].set_title("Crimes par heure")
ax[2].bar(dist["district"].astype(str), dist["r"] * 100); ax[2].set_title("Taux d'arrestation (%) par district")
plt.tight_layout(); plt.savefig(f"{OUT}/exploration.png", dpi=120); plt.close()
log("Exploration -> results/exploration.png, results/crimes_par_type.csv")

# ---------- 4. Classification : arrestation ----------
ratio = df.agg(F.avg("arrest")).first()[0]
data = df.withColumn("weight", F.when(F.col("arrest") == 1, 1 - ratio).otherwise(ratio))
train, test = data.randomSplit([0.8, 0.2], seed=42)
train.cache(); test.cache()

cat = ["primary_type", "location_description"]
num = ["hour", "day_of_week", "month", "is_weekend", "district", "domestic", "latitude", "longitude"]
prep = ([StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat] +
        [OneHotEncoder(inputCols=[c + "_idx" for c in cat], outputCols=[c + "_ohe" for c in cat],
                       handleInvalid="keep"),
         VectorAssembler(inputCols=[c + "_ohe" for c in cat] + num, outputCol="features",
                         handleInvalid="skip")])

auc = BinaryClassificationEvaluator(labelCol="arrest")
models = {
    "LogisticRegression": LogisticRegression(labelCol="arrest", weightCol="weight", maxIter=50, regParam=0.01),
    "RandomForest": RandomForestClassifier(labelCol="arrest", weightCol="weight", numTrees=50,
                                           maxDepth=10, maxBins=64, seed=42),
}
rows, fitted = [], {}
for name, est in models.items():
    log(f"Entraînement {name}...")
    t = time.time()
    m = Pipeline(stages=prep + [est]).fit(train)
    secs = time.time() - t
    pred = m.transform(test)
    r = {"model": name, "AUC": auc.evaluate(pred), "train_seconds": round(secs, 1)}
    for k in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        r[k] = MulticlassClassificationEvaluator(labelCol="arrest", metricName=k).evaluate(pred)
    rows.append(r); fitted[name] = (m, pred)
    m.write().overwrite().save(f"models/{'lr' if name.startswith('Log') else 'rf'}_arrest")
    log(f"  {name}: AUC={r['AUC']:.4f}  accuracy={r['accuracy']:.4f}  ({secs:.0f}s)")

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(f"{OUT}/metrics.csv", index=False)

rf_model, rf_pred = fitted["RandomForest"]
cm = rf_pred.groupBy("arrest").pivot("prediction").count().orderBy("arrest").toPandas()
cm.to_csv(f"{OUT}/confusion_matrix_rf.csv", index=False)

attrs = rf_pred.schema["features"].metadata["ml_attr"]["attrs"]
names = {a["idx"]: a["name"] for g in attrs.values() for a in g}
imp = rf_model.stages[-1].featureImportances.toArray()
fi = (pd.DataFrame({"feature": [names.get(i, str(i)) for i in range(len(imp))], "importance": imp})
        .sort_values("importance", ascending=False).head(15))
fi.to_csv(f"{OUT}/feature_importance.csv", index=False)
ax = fi.plot.barh(x="feature", y="importance", figsize=(9, 6), legend=False)
ax.invert_yaxis(); plt.title("Top 15 variables (Random Forest)"); plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=120); plt.close()

# ---------- 5. Clustering : zones chaudes ----------
log("KMeans...")
geo = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="features").transform(df)
scores = {k: ClusteringEvaluator().evaluate(KMeans(k=k, seed=42).fit(geo).transform(geo))
          for k in [4, 6, 8, 10]}
best_k = max(scores, key=scores.get)
km = KMeans(k=best_k, seed=42).fit(geo)
km.write().overwrite().save("models/kmeans_geo")
clustered = km.transform(geo)
(clustered.groupBy("prediction").agg(F.count("*").alias("crimes"),
                                     F.round(F.avg("arrest") * 100, 1).alias("arrest_pct"))
          .orderBy("prediction").toPandas().to_csv(f"{OUT}/clusters.csv", index=False))
pts = clustered.select("latitude", "longitude", "prediction").sample(min(1.0, 50000 / n), seed=1).toPandas()
c = km.clusterCenters()
plt.figure(figsize=(7, 9))
plt.scatter(pts["longitude"], pts["latitude"], c=pts["prediction"], s=1, cmap="tab10", alpha=0.5)
plt.scatter([x[1] for x in c], [x[0] for x in c], c="black", marker="x", s=150)
plt.title(f"Zones de criminalité — KMeans (k={best_k})")
plt.savefig(f"{OUT}/kmeans_map.png", dpi=120); plt.close()

# ---------- 6. Résumé ----------
summary = {"rows": n, "sample": SAMPLE, "arrest_rate_pct": round(ratio * 100, 2),
           "models": rows, "kmeans_silhouette": scores, "kmeans_best_k": best_k}
with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n================ RÉSULTATS ================")
print(metrics.to_string(index=False))
print(f"KMeans k={best_k}  silhouette={scores[best_k]:.4f}")
print("Fichiers : results/  |  Modèles : models/")
spark.stop()

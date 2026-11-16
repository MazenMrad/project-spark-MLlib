# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PySpark 3.5 / MLlib project on the Chicago Crimes dataset (school project, French comments and output). It predicts whether a crime leads to an arrest (Logistic Regression and Random Forest) and finds crime hot spots (KMeans on latitude/longitude). There is no build step, no linter and no test suite.

## Environment (Windows)

- Spark 3.5 needs **Java 17** (not Java 25) and **Python 3.11** (not 3.13). On Windows it also needs `winutils.exe` + `hadoop.dll` (Hadoop 3.3).
- Portable tools live in `.tools/` (git-ignored): `.tools/jdk-17*` and `.tools/hadoop/bin`. The venv is `.venv/`.
- `env.ps1` sets `JAVA_HOME`, `HADOOP_HOME`, `PYSPARK_PYTHON` and `PATH` from those folders. It must be **dot-sourced**: `. .\env.ps1`.
- `predict.py` sets the same variables itself, so it runs without `env.ps1`. `train.py` does not: run it after `. .\env.ps1`.

Setup: `py -3.11 -m venv .venv`, then `.venv\Scripts\activate`, then `pip install -r requirements.txt`.

## Commands (PowerShell, from the repo root)

```powershell
. .\env.ps1                                     # always first
spark-submit --driver-memory 8g train.py        # full data (~1M rows, ~10 min)
spark-submit --driver-memory 8g train.py 0.1    # 10 % sample (~1 min), use this while developing
spark-submit --driver-memory 8g train.py --api 5000   # download 5000 rows from the Socrata API, no local CSV needed
python predict.py                               # load models/rf_arrest and predict 5 hard-coded crimes
.\history.ps1                                   # Spark History Server on http://localhost:18080 (reads spark-events/)
jupyter notebook                                # chicago_crime_mllib.ipynb
```

Spark UI of a running job: http://localhost:4040.

## Architecture

There are two versions of the same pipeline:

- `chicago_crime_mllib.ipynb`: the interactive prototype. Cell 4 chooses the data source (`SOURCE = "csv"` or `"api"`). It also has an optional tuning step (`RUN_TUNING`, `TrainValidationSplit`) that `train.py` does not have.
- `train.py`: the same steps as one batch job for `spark-submit`. The sections are numbered: 1 load, 2 clean, 3 explore, 4 classify, 5 cluster, 6 summary.

Data flow:

1. **Load**: `data/Crimes.csv` is converted **once** to `data/crimes.parquet`. Later runs read the Parquet file. If you change the CSV loading or the column normalisation (`norm()`: lowercase, spaces become `_`), delete the Parquet folder so it is rebuilt. With `--api N`, data goes to `data/api_sample.csv` and Parquet is skipped.
2. **Clean**: dates use two formats (`MM/dd/yyyy hh:mm:ss a` for the CSV, ISO for the API), so they are parsed with `coalesce(try_to_timestamp(...))`. Rows outside the Chicago bounding box are removed. Time features (`hour`, `day_of_week`, `month`, `is_weekend`) are derived here.
3. **Classify**: arrests are a minority class, so a `weight` column (inverse class ratio) is passed as `weightCol`. Features: `StringIndexer` → `OneHotEncoder` on `primary_type` and `location_description`, plus numeric columns, then `VectorAssembler`. The whole thing is saved as a `PipelineModel`.
4. **Cluster**: KMeans with k in {4, 6, 8, 10}; the best silhouette score wins.

Outputs: `models/lr_arrest`, `models/rf_arrest`, `models/kmeans_geo` (git-ignored), and CSV/PNG/`summary.json` files in `results/` (**not** in `.gitignore`). Event logs go to `spark-events/` (git-ignored) for the History Server.

**Important coupling**: `predict.py` builds input rows by hand with the raw feature columns (`primary_type`, `location_description`, `hour`, `day_of_week`, `month`, `is_weekend`, `district`, `domestic`, `latitude`, `longitude`). If you change the `cat` / `num` feature lists in `train.py` (or in the notebook), update `predict.py` too, and retrain before running it.

`docs/explication.html` is a standalone explainer page (in French) about the project. It is not used by the code.

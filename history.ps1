# Lance le Spark History Server : revoir les anciens runs de train.py sans réentraîner
# Usage :  .\history.ps1   puis ouvrir http://localhost:18080   (Ctrl+C pour arrêter)
. "$PSScriptRoot\env.ps1"
New-Item -ItemType Directory -Force "$PSScriptRoot\spark-events" | Out-Null
$dir = ([System.Uri](Resolve-Path "$PSScriptRoot\spark-events").Path).AbsoluteUri
$env:SPARK_HISTORY_OPTS = "-Dspark.history.fs.logDirectory=$dir"
$env:SPARK_HOME = (& python -c "import pyspark,os;print(os.path.dirname(pyspark.__file__))")
Write-Host "History Server -> http://localhost:18080" -ForegroundColor Green
& "$env:SPARK_HOME\bin\spark-class.cmd" org.apache.spark.deploy.history.HistoryServer

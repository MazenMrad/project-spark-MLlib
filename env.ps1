# Active l'environnement Spark du projet (Java 17 + winutils + venv Python 3.11)
# Usage :  . .\env.ps1     (avec le point devant !)
$root = $PSScriptRoot
$jdk = Get-ChildItem "$root\.tools" -Directory -Filter "jdk-17*" | Select-Object -First 1
$env:JAVA_HOME      = $jdk.FullName
$env:HADOOP_HOME    = "$root\.tools\hadoop"
$env:PYSPARK_PYTHON = "$root\.venv\Scripts\python.exe"
$env:PATH = "$root\.venv\Scripts;$env:JAVA_HOME\bin;$env:HADOOP_HOME\bin;$env:PATH"
Write-Host "Spark env OK  (Java: $($jdk.Name))" -ForegroundColor Green

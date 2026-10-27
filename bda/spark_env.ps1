# Save as: D:\Projects-Battleground\Sentiment_Analysis\spark_env.ps1

$env:JAVA_HOME = "C:\Java\jdk-17"
$env:HADOOP_HOME = "C:\hadoop"
$env:SPARK_HOME = "D:\Projects-Battleground\Sentiment_Analysis\.venv\Lib\site-packages\pyspark"
$env:PATH = "C:\Java\jdk-17\bin;$env:SPARK_HOME\bin;$env:HADOOP_HOME\bin;" + $env:PATH
$env:PYSPARK_PYTHON = "D:\Projects-Battleground\Sentiment_Analysis\.venv\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON = "D:\Projects-Battleground\Sentiment_Analysis\.venv\Scripts\python.exe"

Write-Host "✅ Spark environment ready" -ForegroundColor Green
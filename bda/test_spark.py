from pyspark import SparkContext

# No findspark.init() needed

try:
    sc = SparkContext("local", "TestApp")
    print("Spark version:", sc.version)
    print("Spark works! The environment is configured correctly.")
    sc.stop()
except Exception as e:
    print(f"An error occurred: {e}")
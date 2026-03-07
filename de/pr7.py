#Prac 7A: Write simple pyspark driver program
!pip install pyspark==3.5.0
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
spark=SparkSession.builder.appName('myapp').getOrCreate()
sc=spark.sparkContext
rdd=sc.parallelize([1,2,3,4,5])
rdd_squared=rdd.map(lambda x:x*x)
rdd_squared.collect()
spark.stop()

#Prac 7B: Working with different transformations and actions on RDD by fetching data from external file

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

spark = SparkSession.builder.getOrCreate()

sentence = spark.read.format("csv").option("header",True).option("inferSchema",True).load("./test_news.csv")

sentence = sentence.withColumn("id", monotonically_increasing_id())
sentence = sentence.withColumn("label", sentence.id*0)

sentence = sentence.filter(sentence.id<2200)
sentence.count()
sentenceData = sentence.select("id","label","text")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
wordsData.show(5)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)
featurizedData.show(10)
featurizedData.printSchema()

featurizedData.cache()
#idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = IDF(inputCol="rawFeatures", outputCol="features").fit(featurizedData)
#idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

dataset = rescaledData.select("features")
from pyspark.ml.clustering import KMeans
# Trains a k-means model.
kmeans = KMeans().setK(10).setSeed(1)
model = kmeans.fit(dataset)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
        print(center)

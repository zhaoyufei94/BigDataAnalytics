from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
spark = SparkSession.builder.getOrCreate()

data = spark.read.format("csv").option("header",True).option("inferSchema",True).load("./pima_indian.txt")

from pyspark.ml.feature import VectorAssembler
label = ["label"]
assembler = VectorAssembler(
            inputCols=[x for x in data.columns if x not in label],
                outputCol='features')
data = assembler.transform(data)
data = data.select('label','features')

# Split the data into train and test
splits = data.select("label", "features").randomSplit([0.8, 0.2], 1234)
train = splits[1]
test = splits[0]

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rf = RandomForestClassifier()

model = rf.fit(train)

predictions = model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                             predictionCol="prediction",
                                            metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy of RandomForest= " + str(accuracy))

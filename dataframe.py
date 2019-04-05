import pyspark.sql.types 
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import  StringIndexer, OneHotEncoder,VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.functions import udf,col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SQLContext

global Path    
Path="file:/home/vivid/Downloads/"
def CreateSparkContext():
    def SetLogger( sc ):
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
        logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
        logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    

    sparkConf = SparkConf().setAppName("RunDecisionTreeBinary").set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print(("master="+sc.master))    
    SetLogger(sc)
    return (sc)
sc = CreateSparkContext()
print("read data")
sqlContext = SQLContext(sc)
row_df = sqlContext.read.format("csv").option("header", "true").load(Path+"data/hour.csv")
df= row_df.drop("instant").drop("dteday") \
                            .drop('yr').drop("casual").drop("registered")
hour_df= df.select([ col(column).cast("double").alias(column) for column in df.columns])

train_df, test_df = hour_df.randomSplit([0.7, 0.3])
train_df.cache()
test_df.cache

from pyspark.ml import Pipeline
from pyspark.ml.feature import  StringIndexer,  VectorIndexer,VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

print("setup pipeline")
featuresCols = hour_df.columns[:-1]
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="aFeatures")
vectorIndexer = VectorIndexer(inputCol="aFeatures", outputCol="features", maxCategories=24)
dt = DecisionTreeRegressor(labelCol="cnt",featuresCol= 'features',maxDepth=10,impurity="variance",maxBins=100)
dt_pipeline = Pipeline(stages=[vectorAssembler,vectorIndexer ,dt])

print("train model")
dt_pipelineModel = dt_pipeline.fit(train_df)
print("predict")
predicted = dt_pipelineModel.transform(test_df).select('season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', \
                     'weathersit', 'temp', 'atemp', 'hum', 'windspeed','cnt','prediction').show(10)
print(predicted)

print("eval model")
evaluator = RegressionEvaluator(labelCol='cnt', predictionCol='prediction', metricName="rmse")
predictions = dt_pipelineModel.transform(test_df)
auc = evaluator.evaluate(predictions)
print(auc)

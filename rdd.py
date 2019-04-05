from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.evaluation import BinaryClassificationMetrics

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

trainLine = []
validLine = []
categoriesMap = []

print("read data")
rawDataWithHeader = sc.textFile(Path+"data/hour.csv")
header = rawDataWithHeader.first() 
rawData = rawDataWithHeader.filter(lambda x:x !=header)  
#rawData.take(5)
rawData = rawData.map(lambda x: x.split(","))
rawData.take(2)

lines = rawData.map(lambda r : (r[2],r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13],r[16]))
lines.take(5)

def extract_label(record):
    label=(record[-1])
    return float(label)

def extract_features(field,featureEnd):
    Features = (field[0:featureEnd])
    return  Features

(trainLine, validLine) = lines.randomSplit([7, 3])
trainData = trainLine.map( lambda r:LabeledPoint( extract_label(r), extract_features(r,len(r) - 1)))
validationData = validLine.map( lambda r:LabeledPoint( extract_label(r), extract_features(r,len(r) - 1)))
trainData.take(1)

from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

print("train model")
model = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo={},
                                    impurity='variance', maxDepth=10, maxBins=100)
print("predict")
dataRDD = validLine.map(lambda r: ( r[:] ,extract_features(r,len(r)-1)))
for data in dataRDD.take(10):
    predictResult = model.predict(data[1])
    print(" season：" +str(data[0][0]) + 
          "  mnth：" +str(data[0][1]) + 
          "  hr：" +str(data[0][2]) + 
          "  holiday：" +str(data[0][3]) + 
          "  week：" +str(data[0][4]) + 
          "  workday：" +str(data[0][5]) + 
          "  weathersit：" +str(data[0][6]) + 
          "  temp：" +str(data[0][7]) + 
          "  atemp：" +str(data[0][8]) + 
          "  hum：" +str(data[0][9]) + 
          "  windspeed：" +str(data[0][10]) + 
          "  cnt：" +str(data[0][11]) +      
          "  pridiction:"+ str(predictResult))
    
print("eval model")    
def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLabels)
    # Root mean squared error
    RMSE = metrics.rootMeanSquaredError
    return RMSE

RMSE = evaluateModel(model, validationData)
print("RMSE : " + str(RMSE))

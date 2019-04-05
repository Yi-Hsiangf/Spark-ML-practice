import org.apache.log4j.Logger

import org.apache.log4j.Level

import org.apache.spark.storage.StorageLevel

import org.apache.spark.ml.classification.DecisionTreeClassifier

import org.apache.spark.ml.Pipeline

import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.functions.col

import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoderEstimator,VectorAssembler}

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark.ml.feature.VectorIndexer

import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.regression.DecisionTreeRegressionModel

import org.apache.spark.ml.regression.DecisionTreeRegressor


object RunDecisionTreeBinary {

    case class Data(instant: Double,

                    dtday: String,

                    season: Double,

                    yr: Double,

                    mnth: Double,

                    hr: Double,

                    holiday: Double,

                    weekday: Double,

                    workingday: Double,

                    weathersit: Double,

                    temp: Double,

                    atemp: Double,

                    hum: Double,

                    windspeed: Double,

                    casual: Double,

                    registered: Double,

                    cnt: Double)

  def main(args: Array[String]): Unit = {

    SetLogger()

    val spark = SparkSession.builder().appName("Spark SQL basic example").master("local[4]").config("spark.ui.showConsoleProgress","false").getOrCreate()

    import spark.implicits._

    val sch = org.apache.spark.sql.Encoders.product[Data].schema

    println("read data")

    val ds = spark.read.format("csv").option("header", "true").schema(sch).load("file:/home/vivid/Downloads/data/hour.csv").as[Data]

    val data = ds.randomSplit(Array(0.7,0.3))

    val train_ds = data(0).select("season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt")

    val test_ds = data(1).select("season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "cnt")

	//println("show test_ds")
	//test_ds.show(10)
    println("setup pipeline")

    val vectorAssembler = new VectorAssembler().setInputCols(Array("season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed")).setOutputCol("aFeatures")

    val vectorIndexer = new VectorIndexer().setInputCol("aFeatures").setOutputCol("features").setMaxCategories(24)
	
	val dt = new DecisionTreeRegressor().setLabelCol("cnt").setFeaturesCol("features").setMaxDepth(10).setImpurity("variance").setMaxBins(100)

    val pipeline = new Pipeline().setStages(Array(vectorAssembler,vectorIndexer,dt))

    println("train model")

    val pipelineModel = pipeline.fit(train_ds)

    print("predict")

    val predicted=pipelineModel.transform(test_ds).select("season","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","cnt","prediction").show(10)

    println(predicted)

    println("eval model")

	val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")

    val predictions =pipelineModel.transform(test_ds)

    val rmse= evaluator.evaluate(predictions)

    println(rmse)

  }



  def SetLogger() = {

    Logger.getLogger("org").setLevel(Level.OFF)

    Logger.getLogger("com").setLevel(Level.OFF)

    System.setProperty("spark.ui.showConsoleProgress", "false")

    Logger.getRootLogger().setLevel(Level.OFF);

  }

}

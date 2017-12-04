package SparkSQL.BigDataClass
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.log4j.{Level, Logger}


/**
 * @author ${user.name}
 */
object App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  
  def main(args : Array[String]) {
     val spark = SparkSession.builder().appName("BigDataProject").master("local").getOrCreate()

    val df = spark.read
      .option("header", true)
      .csv("C:/Users/tjass/Documents/Big_Data/Projet_Spark_ML/2006.csv")
      .drop("Year", "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay","TailNum", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
      .drop( "CancellationCode")
      .drop("DepTime", "CRSArrTime")
      .withColumn("CRSElapsedTime", col("CRSElapsedTime").cast("Double"))
      .withColumn("DepDelay", col("DepDelay").cast("Double"))
      .withColumn("Distance", col("Distance").cast("Double"))
      .withColumn("TaxiOut", col("TaxiOut").cast("Double"))
    val df2 = df.filter(df.col("Cancelled") < 1).filter(df.col("ArrDelay") =!= "NA").drop("Cancelled")
    val take2First = udf { (CRSDepTime: String) =>
      if (CRSDepTime.length < 3) "00" else if (CRSDepTime.length == 3) CRSDepTime.substring(0, 1) else CRSDepTime.substring(0, 2)
    }
    val df1 = df2.withColumn("CRSDepTime", take2First(df2("CRSDepTime")))

    df1.createOrReplaceTempView("Delay")


    //Calculation of the mean of delay per hour

    val mean = spark.sql("SELECT CRSDepTime AS CRSDepTime1, MEAN(ArrDelay) AS mean_delay_per_hour FROM Delay GROUP BY CRSDepTime")
    val df3 = df1.join(mean, df1("CRSDepTime") === mean("CRSDepTime1") ).drop("CRSDepTime1")

    val indexer1 = new StringIndexer().setInputCol("UniqueCarrier").setOutputCol("UCIndex")

    val indexer2 = new StringIndexer()
      .setInputCol("Month")
      .setOutputCol("MonthIndex")

    val indexer3 = new StringIndexer()
      .setInputCol("DayofMonth")
      .setOutputCol("DayOfMonthIndex")

    val indexer4 = new StringIndexer()
      .setInputCol("DayOfWeek")
      .setOutputCol("DayOfWeekIndex")

    val indexer5 = new StringIndexer()
      .setInputCol("CRSDepTime")
      .setOutputCol("CRSDepTimeIndex")

    val indexer6 = new StringIndexer()
      .setInputCol("FlightNum")
      .setOutputCol("FlightNumIndex")

    val indexer7 = new StringIndexer()
      .setInputCol("Origin")
      .setOutputCol("OriginIndex")

    val indexer8 = new StringIndexer()
      .setInputCol("Dest")
      .setOutputCol("DestIndex")
    val indexed1 = indexer1.fit(df3).transform(df3)
    val indexed2 = indexer2.fit(indexed1).transform(indexed1)
    val indexed3 = indexer3.fit(indexed2).transform(indexed2)
    val indexed4 = indexer4.fit(indexed3).transform(indexed3)
    val indexed5 = indexer5.fit(indexed4).transform(indexed4)
    val indexed6 = indexer6.fit(indexed5).transform(indexed5)
    val indexed7 = indexer7.fit(indexed6).transform(indexed6)
    val indexed = indexer8.fit(indexed7).transform(indexed7)

    indexed.printSchema()

    //ArrDelay placé à la fin (attention pas echangé mais après CRSElapsedTime il y a DepDelay
    val rdd_ml = indexed.rdd.map((x:Row) => {
      Row(x.getAs[Double](14),x.getAs[Double](15),x.getAs[Double](16),x.getAs[Double](17),x.getAs[Double](13),x.getAs[Double](6),x.getAs[Double](8),x.getAs[Double](19),x.getAs[Double](20),x.getAs[Double](11),x.getAs[Double](12),x.getAs[Double](13), x.getAs[Double](7))
    })

    val labeledPoints = rdd_ml.map((x:Row) => LabeledPoint(x.toSeq.last.toString.toDouble,Vectors.dense(x.toSeq.init.toArray.map(_.toString.toDouble))))


    val numofCRSDepTime = df1.select("CRSDepTime").distinct.count.toInt
    val numOfUC = df1.select("UniqueCarrier").distinct.count.toInt
    val numofOriginDest = df1.select("Origin").distinct.count.toInt

    val featureSubsetStrategy = "auto"
    val impurity = "variance"
    val maxDepth = 12
    val maxBins = 2700
    val seed = 5043

    val categoricalFeaturesInfo: Map[Int, Int] = Map((0,12),(1,31),(2,7),(3,numofCRSDepTime),(4,numOfUC),(7,numofOriginDest),(8,numofOriginDest))
    val splits = labeledPoints.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val model = RandomForest.trainRegressor(trainingData,categoricalFeaturesInfo,100,featureSubsetStrategy, impurity, maxDepth, maxBins,seed)

    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)

  }
}


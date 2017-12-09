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
import org.apache.spark.sql.functions.stddev_pop
import org.apache.spark.sql.functions.avg
import org.apache.spark.mllib.stat.Statistics

/**
 * @author ${user.name}
 */
object App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  
  
  def main(args : Array[String]) {
    
    if(args.length != 1){
      throw new Exception("You need to specify the path as parameter")
    }
    
     val spark = SparkSession.builder().appName("BigDataProject").master("local").getOrCreate()
     

    val df = spark.read
      .option("header", true)
      .csv(args(0))
      .drop("Year", "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn","FlightNum", "Diverted", "CarrierDelay","TailNum", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay")
      .drop( "CancellationCode")
      .drop("DepTime", "CRSArrTime")
    
      
    //FlightNum = Origin + Dest + Hour of departure
    val df2 = df.filter(df.col("ArrDelay") =!= "NA")
                .filter(df.col("Month") =!= "NA")
                .filter(df.col("DayofMonth") =!= "NA")
                .filter(df.col("DayOfWeek") =!= "NA")
                .filter(df.col("CRSDepTime") =!= "NA")
                .filter(df.col("UniqueCarrier") =!= "NA")
                .filter(df.col("CRSElapsedTime") =!= "NA")
                .filter(df.col("DepDelay") =!= "NA")
                .filter(df.col("Origin") =!= "NA")
                .filter(df.col("Dest") =!= "NA")
                .filter(df.col("Distance") =!= "NA")
                .filter(df.col("TaxiOut") =!= "NA")
                .filter(df.col("Cancelled") =!= "NA")
                .filter(df.col("Cancelled") < 1)
                .drop("Cancelled")
   
   
    val take2First = udf { (CRSDepTime: String) =>
      if (CRSDepTime.length < 3) "00" else if (CRSDepTime.length == 3) CRSDepTime.substring(0, 1) else CRSDepTime.substring(0, 2)
    }
    val df1 = df2.withColumn("CRSDepTime", take2First(df2("CRSDepTime")))
     
    
    
    df1.createOrReplaceTempView("Delay")


    //Calculation of the mean of delay per hour

    val mean = spark.sql("SELECT CRSDepTime AS CRSDepTime1, MEAN(ArrDelay) AS mean_delay_per_hour FROM Delay GROUP BY CRSDepTime")
    val df3 = df1.join(mean, df1("CRSDepTime") === mean("CRSDepTime1") ).drop("CRSDepTime1")

    val std_distance = df3.select(stddev_pop(df3.col("Distance"))).first.getDouble(0)
    val std_arrdelay = df3.select(stddev_pop(df3.col("ArrDelay"))).first.getDouble(0)
    val std_depdelay = df3.select(stddev_pop(df3.col("DepDelay"))).first.getDouble(0)
    val std_taxiout = df3.select(stddev_pop(df3.col("TaxiOut"))).first.getDouble(0)
    val std_meandelay = df3.select(stddev_pop(df3.col("mean_delay_per_hour"))).first.getDouble(0)
    
    val avg_distance = df3.select(avg(df3.col("Distance"))).first.getDouble(0)
    val avg_arrdelay = df3.select(avg(df3.col("ArrDelay"))).first.getDouble(0)
    val avg_depdelay = df3.select(avg(df3.col("DepDelay"))).first.getDouble(0)
    val avg_taxiout = df3.select(avg(df3.col("TaxiOut"))).first.getDouble(0)
    val avg_meandelay = df3.select(avg(df3.col("mean_delay_per_hour"))).first.getDouble(0)
    
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
      .setInputCol("Origin")
      .setOutputCol("OriginIndex")

    val indexer7 = new StringIndexer()
      .setInputCol("Dest")
      .setOutputCol("DestIndex")
      
    val indexed1 = indexer1.fit(df3).transform(df3)
    val indexed2 = indexer2.fit(indexed1).transform(indexed1)
    val indexed3 = indexer3.fit(indexed2).transform(indexed2)
    val indexed4 = indexer4.fit(indexed3).transform(indexed3)
    val indexed5 = indexer5.fit(indexed4).transform(indexed4)
    val indexed6 = indexer6.fit(indexed5).transform(indexed5)
    val indexed = indexer7.fit(indexed6).transform(indexed6)

   
    //ArrDelay placé à la fin (attention pas echangé mais après CRSElapsedTime il y a DepDelay
    val rdd_ml = indexed.rdd.map((x:Row) => {
      val month = x.getAs[Double](14);
      val day_of_month = x.getAs[Double](15);
      val day_of_week = x.getAs[Double](16);
      val CRSDepTime = x.getAs[Double](17);
      val UC = x.getAs[Double](13);
      val CRSElapsedTime = x.getAs[String](5).toDouble;
      val DepDelay = x.getAs[String](7).toDouble;
      val Origin = x.getAs[Double](18);
      val Dest = x.getAs[Double](19);
      val Distance = x.getAs[String](10).toDouble;
      val TaxiOut = x.getAs[String](11).toDouble;
      val mean_delay_per_hour = x.getAs[Double](12);
      val ArrDelay = x.getAs[String](6).toDouble;
      Row(month,day_of_month,day_of_week,CRSDepTime,UC,CRSElapsedTime,(DepDelay-avg_depdelay)/std_depdelay,Origin,Dest,(Distance-avg_distance)/std_distance,(TaxiOut-avg_taxiout)/std_taxiout,(mean_delay_per_hour-avg_meandelay)/std_meandelay, (ArrDelay-avg_arrdelay)/std_arrdelay)
    })
    
    
    
    val labeledPoints = rdd_ml.map((x:Row) => LabeledPoint(x.toSeq.last.toString.toDouble,Vectors.dense(x.toSeq.init.toArray.map(_.toString.toDouble))))


    val numofCRSDepTime = df1.select("CRSDepTime").distinct.count.toInt
    val numOfUC = df1.select("UniqueCarrier").distinct.count.toInt
    val numofOriginDest = df1.select("Origin").distinct.count.toInt
    

    val featureSubsetStrategy = "auto"
    val impurity = "variance"
    val maxDepth = 9
    val maxBins = 2700
    val seed = 5043

    val categoricalFeaturesInfo: Map[Int, Int] = Map((0,12),(1,31),(2,7),(3,numofCRSDepTime),(4,numOfUC),(7,numofOriginDest),(8,numofOriginDest))
    val splits = labeledPoints.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    println("Starting RandomForest");
    val model = RandomForest.trainRegressor(trainingData,categoricalFeaturesInfo,50,featureSubsetStrategy, impurity, maxDepth, maxBins,seed)
    println("Finish RandomForest")
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    
    labelsAndPredictions.take(20).foreach(println)
    
    
    val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
    val testRMSE = Math.sqrt(testMSE)
    val MSE = testMSE * std_arrdelay + avg_arrdelay
    val RMSE = Math.sqrt(MSE)
    val r2 = Statistics.corr(labelsAndPredictions.map(_._2), labelsAndPredictions.map(_._1), "pearson")
    
    println("Test Normalized Mean Squared Error = " + testMSE)
    println("Test Normalized Root Mean Squared Error = " + testRMSE)
    println("Test Mean Squared Error = " + MSE)
    println("Test Root Mean Squared Error = " + RMSE)
    println("Test Pearson Correlation Coefficient = " + r2)

  }
}


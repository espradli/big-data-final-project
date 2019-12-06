package chicagotransportation

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import swiftvis2.plotting.Plot
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.SwingRenderer
import swiftvis2.plotting.ColorGradient
import shapeless.union
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object ChicagoTransportation{
  def main(args: Array[String]): Unit ={
    val spark = SparkSession.builder()
      //.appName("ChicagoTransportation").getOrCreate()
      .master("local[*]").appName("ChicagoTransportation").getOrCreate()
    import spark.implicits._
    
    spark.sparkContext.setLogLevel("WARN")

    def makeRandomSample(): Unit = {
      lazy val taxiData = spark.read.option("inferSchema", true)
        .option("header", "true")
        .csv("/data/BigData/students/espradli/taxiData.csv")

      lazy val randomSample = taxiData.randomSplit(Array(.05, .95))(0)
      
      randomSample.write
        //.format("com.databricks.spark.csv").option("header", "true").save("/users/espradli/BigData/taxiDataRandom.csv")
        //.option("header", "true")
        .csv("/data/BigData/students/espradli/taxiDataRandom.csv")
    }

    lazy val taxiData = spark.read.option("inferSchema", true)
        .option("header", "true")
        .csv("/Users/emersonspradling/ChicagoTransport/taxiDataShort.csv")

    lazy val heatMap = {
      //? Possibly made smaller values bigger in size on map and vise versa to make everything bleed together
      val pickupData = taxiData.select("Pickup Centroid Latitude","Pickup Centroid Longitude")
        .withColumnRenamed("Pickup Centroid Latitude", "lat")
        .withColumnRenamed("Pickup Centroid Longitude", "long")
      val dropoffData = taxiData.select("Dropoff Centroid Latitude","Dropoff Centroid Longitude")
        .withColumnRenamed("Dropoff Centroid Latitude", "lat")
        .withColumnRenamed("Dropoff Centroid Longitude", "long")
      val unionData = pickupData.union(dropoffData).where($"lat" > 35 && $"long" < -80)
        .groupBy("lat", "long").agg(count("*").as("count")).collect()
      
      val maxCount = unionData.map(_.getAs[Long]("count")).max

      val x = unionData.map(_.getAs[Double]("long"))
      val y = unionData.map(_.getAs[Double]("lat"))
      val cg = ColorGradient(1.0 -> CyanARGB, (.05*maxCount) -> BlueARGB, (.97*maxCount) -> RedARGB) 
      val color = unionData.map(c => cg(c.getAs[Long]("count")))
      val size = unionData.map(c => ((c.getAs[Long]("count") * 20).toDouble / maxCount) + 4)

      val plot = Plot.scatterPlot(x, y, "Taxi Activity Heat Map", "Longitude", "Latitude", size, color)
      SwingRenderer(plot, 800, 800, true)
    }

    lazy val cluster = {
      val va = new VectorAssembler()
        .setInputCols(Array("Trip Miles", "Fare", "Trip Seconds"))
        .setOutputCol("oldFeatures")
      val taxiDataWithFeature = va.setHandleInvalid("skip").transform(taxiData)

      val scaler = new StandardScaler()
        .setInputCol("oldFeatures")
        .setOutputCol("features")
      val scalerModel = scaler.fit(taxiDataWithFeature)
      val taxiDataScaled = scalerModel.transform(taxiDataWithFeature).select($"Pickup Centroid Latitude", $"Pickup Centroid Longitude", $"Trip Miles", $"Fare", $"Trip Seconds", $"features")

      val kmeans = new KMeans().setK(5)
      val kmeansModel = kmeans.fit(taxiDataScaled)
      val taxiDataWithClusters = kmeansModel.transform(taxiDataScaled)

      println(s"Test Set Accuracy = ${new ClusteringEvaluator().evaluate(taxiDataWithClusters)}")

      val clusterAVGSeconds = taxiDataWithClusters.groupBy("prediction").agg(mean("Trip Seconds")).show
      val clusterAVGTotal = taxiDataWithClusters.groupBy("prediction").agg(mean("Fare")).show
      val clusterAVGTravel = taxiDataWithClusters.groupBy("prediction").agg(mean("Trip Miles")).show

      val predictionCount = taxiDataWithClusters.groupBy("Pickup Centroid Latitude", "Pickup Centroid Longitude", "prediction").agg(count("*").as("predictionCount"))
      val maxPredictionCount =  predictionCount.groupBy("Pickup Centroid Latitude", "Pickup Centroid Longitude").agg(max("predictionCount").as("max"))
      val pointsWithPrediction = predictionCount.join(maxPredictionCount, Seq("Pickup Centroid Latitude", "Pickup Centroid Longitude")).filter($"max" - $"predictionCount" === 0)
      
      val pointsCount =  taxiDataWithClusters.groupBy($"Pickup Centroid Latitude", $"Pickup Centroid Longitude").agg(count("*").as("pointCount"))

      val pointsWithCountPrediction = pointsWithPrediction.join(pointsCount, Seq("Pickup Centroid Latitude", "Pickup Centroid Longitude")).collect()
      val maxCount = pointsWithCountPrediction.map(_.getAs[Long]("pointCount")).max

      val x = pointsWithCountPrediction.map(_.getAs[Double]("Pickup Centroid Longitude"))
      val y = pointsWithCountPrediction.map(_.getAs[Double]("Pickup Centroid Latitude"))
      val cg = ColorGradient(0.0 -> BlackARGB, 1.0 -> GreenARGB, 2.0 -> RedARGB, 3.0 -> BlueARGB, 4.0 -> MagentaARGB) 
      val color = pointsWithCountPrediction.map(c => cg(c.getAs[Int]("prediction")))
      val size = pointsWithCountPrediction.map(c => ((c.getAs[Long]("pointCount") * 20).toDouble / maxCount) + 4)

      val plot = Plot.scatterPlot(x, y, "Taxi Distance Clustering", "Longitude", "Latitude", size, color)
      SwingRenderer(plot, 800, 800, true)
    }

    println(cluster)
  }
}
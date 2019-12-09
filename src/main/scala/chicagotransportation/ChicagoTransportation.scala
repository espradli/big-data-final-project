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
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation

object ChicagoTransportation{
  def main(args: Array[String]): Unit ={
    val spark = SparkSession.builder()
      // .appName("ChicagoTransportation").getOrCreate()
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
      // .csv("/data/BigData/students/espradli/taxiData.csv")
      .csv("/Users/emersonspradling/ChicagoTransport/taxiDataShort.csv")

    // lazy val lstationData = spark.read.option("inferSchema", true)
    //   .option("header", "true")
    //   // .csv("/data/BigData/students/espradli/LStationStops.csv")
    //   .csv("/Users/emersonspradling/ChicagoTransport/LStationStops.csv")

    lazy val heatMap = {
      //TODO? Possibly made smaller values bigger in size on map and vise versa to make everything bleed together
      //TODO insert L train stops
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
      val cg = ColorGradient(1.0 -> GreenARGB, (.01*maxCount) -> BlueARGB, (.97*maxCount) -> RedARGB) 
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
      
      // val pointsCount =  taxiDataWithClusters.groupBy($"Pickup Centroid Latitude", $"Pickup Centroid Longitude").agg(count("*").as("pointCount"))
      // val pointsWithCountPrediction = taxiDataWithClusters.groupBy($"Pickup Centroid Latitude", $"Pickup Centroid Longitude")
      //   .agg(mean("prediction").as("prediction")).join(pointsCount, Seq("Pickup Centroid Latitude", "Pickup Centroid Longitude")).collect()
      // val maxCount = pointsWithCountPrediction.map(_.getAs[Long]("pointCount")).max

      val latlongplot = {
        val x = pointsWithCountPrediction.map(_.getAs[Double]("Pickup Centroid Longitude"))
        val y = pointsWithCountPrediction.map(_.getAs[Double]("Pickup Centroid Latitude"))
        val cg = ColorGradient(0.0 -> YellowARGB, 1.0 -> GreenARGB, 2.0 -> RedARGB, 3.0 -> BlueARGB, 4.0 -> MagentaARGB) 
        val color = pointsWithCountPrediction.map(c => cg(c.getAs[Int]("prediction")))
        val size = pointsWithCountPrediction.map(c => ((c.getAs[Long]("pointCount") * 20).toDouble / maxCount) + 4)

        val plot = Plot.scatterPlot(x, y, "Taxi Distance Clustering", "Longitude", "Latitude", size, color)
        plot
      }

      val milesSecondsFarePlot = {
        val data = taxiDataWithClusters.randomSplit(Array(.05, .95))(1).filter($"Trip Miles" < 300.0 && $"Trip Seconds" < 12000.0).collect

        val x = data.map(_.getAs[Double]("Trip Miles"))
        val y = data.map(_.getAs[Int]("Trip Seconds"))
        val cg = ColorGradient(0.0 -> YellowARGB, 1.0 -> GreenARGB, 2.0 -> RedARGB, 3.0 -> BlueARGB, 4.0 -> MagentaARGB) 
        val color = data.map(c => cg(c.getAs[Int]("prediction")))
        val size = data.map(c => ((c.getAs[Double]("Fare") * 20).toDouble / maxCount) + 4)

        val plot = Plot.scatterPlot(x, y, "Taxi Distance Clustering", "Trip Miles", "Trip Seconds", size, color)
        plot
      }

      SwingRenderer(latlongplot, 800, 800, true)
      SwingRenderer(milesSecondsFarePlot, 800, 800, true)
    }

    lazy val correlationCoefMatrixRegression = {
      val va = new VectorAssembler()
        .setInputCols(taxiData.drop("Taxi ID", "Trip ID","Trip Start Timestamp", "Trip End Timestamp", "Payment Type", "Company", "Pickup Centroid Location", "Dropoff Centroid  Location").columns)
        .setOutputCol("features")

      val taxiDataWithFeature = va.setHandleInvalid("skip").transform(taxiData)
      
      val Row(m: Matrix) = Correlation.corr(taxiDataWithFeature, "features").head
      val lCol = m.colIter.toSeq(7).toArray
      lCol.map(Math.abs).zip(taxiData.drop("Taxi ID", "Trip ID","Trip Start Timestamp", "Trip End Timestamp", "Payment Type", "Company", "Pickup Centroid Location", "Dropoff Centroid  Location").columns)
        .sortBy(-_._1).take(4).foreach(println)
    }

    lazy val regression = {
      val va = new VectorAssembler()
        .setInputCols(Array("Trip Miles", "Trip Seconds"))
        .setOutputCol("features")
      val taxiDataWithFeature = va.setHandleInvalid("skip").transform(taxiData.select("Trip Miles", "Trip Seconds", "Pickup Centroid Longitude", "Tips").filter($"Tips".isNotNull))

      val lr = new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("Tips")

      val lrModel = lr.fit(taxiDataWithFeature)

      val summary = lrModel.summary
      println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
      println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
      println(s"R2: ${summary.r2}")

      val c = taxiData.limit(3000).collect()
      
      val x1 = c.map(_.getAs[Double]("Tips"))
      val y1 = c.map(_.getAs[Double]("Trip Miles"))
      val color1 = BlackARGB
      val size1 = 5

      val x2 = c.map(_.getAs[Double]("Tips"))
      val y2 = c.map(_.getAs[Int]("Trip Seconds"))
      val color2 = RedARGB
      val size2 = 5

      //TODO? possibly extimate tip nulls

      val plot = Plot.scatterPlot(x1, y1, "Trip Miles in Relation to Tips", "Tips", "Trip Miles", size1, color1)
      val plot2 = Plot.scatterPlot(x2, y2, "Trip Seconds in Relation to Tips", "Tips", "Trip Seconds", size2, color2)
      SwingRenderer(plot, 800, 800, true)
      SwingRenderer(plot2, 800, 800, true)
    }

    lazy val correlationCoefMatrixClassification = {
      
      //coppied
      val taxiDataDropNa = taxiData.na.drop()
      
      val paymentIndexer = new StringIndexer()
        .setInputCol("Payment Type")
        .setOutputCol("paymentIndex")

      val companyIndexer = new StringIndexer()
        .setInputCol("Company")
        .setOutputCol("label")

      val pipeline = new Pipeline()
        .setStages(Array(paymentIndexer, companyIndexer))

      val taxiDataTransformed = pipeline.fit(taxiDataDropNa).transform(taxiDataDropNa)
      //
      
      val va = new VectorAssembler()
        .setInputCols(taxiDataTransformed.drop("Taxi ID", "Trip ID","Trip Start Timestamp", "Trip End Timestamp", "Payment Type", "Company", "Pickup Centroid Location", "Dropoff Centroid  Location").columns)
        .setOutputCol("features")

      val taxiDataWithFeature = va.setHandleInvalid("skip").transform(taxiDataTransformed)
      
      val Row(m: Matrix) = Correlation.corr(taxiDataWithFeature, "features").head
      val lCol = m.colIter.toSeq(16).toArray
      lCol.map(Math.abs).zip(taxiDataTransformed.drop("Taxi ID", "Trip ID","Trip Start Timestamp", "Trip End Timestamp", "Payment Type", "Company", "Pickup Centroid Location", "Dropoff Centroid  Location").columns)
        .sortBy(-_._1).take(4).foreach(println)
    }
    
    lazy val classification = {
      val taxiDataDropNa = taxiData.na.drop()
      
      taxiDataDropNa.select("Payment Type").distinct().show()
      val paymentIndexer = new StringIndexer()
        .setInputCol("Payment Type")
        .setOutputCol("label")

      val taxiDataTransformed = paymentIndexer.fit(taxiDataDropNa).transform(taxiDataDropNa)
      
      val va = new VectorAssembler()
        //.setInputCols(Array("Trip Total", "Tips", "Fare", "paymentIndex"))
        .setInputCols(Array("Trip Miles", "Trip Seconds", "Tips", "Extras"))
        .setOutputCol("features")
      val taxiDataWithFeature = va.setHandleInvalid("skip").transform(taxiDataTransformed)

      // Split the data into train and test
      val splits = taxiDataWithFeature.randomSplit(Array(0.7, 0.3))
      val train = splits(0)
      val test = splits(1)

      // specify layers for the neural network:
      // input layer of size 4 (features), two intermediate of size 5 and 4
      // and output of size 3 (classes)
      val layers = Array[Int](4, 10, 5, taxiDataTransformed.select("label").distinct().count.toInt)

      // create the trainer and set its parameters
      val trainer = new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setMaxIter(100)

      // train the model
      val mpcModel = trainer.fit(train)

      // compute accuracy on the test set
      val result = mpcModel.transform(test)
      val predictionAndLabels = result.select("prediction", "label")
      val evaluator = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy")

      println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

      val plottingData = result.limit(50000).select("prediction", "label", "Tips", "Trip Miles").collect()//.sortBy(_.getAs[Double]("label"))
      val zippedPlottingData = plottingData.zip(0 until plottingData.length-1)
      val maxTip = zippedPlottingData.map(c => ((c._1.getAs[Double]("Tips")))).max
      val x1 = zippedPlottingData.map(_._1.getAs[Double]("Trip Miles")).filter(_ < 60)
      val y1 = zippedPlottingData.map(_._1.getAs[Double]("label"))
      val color1 = BlackARGB
      val size1 = zippedPlottingData.map(c => ((c._1.getAs[Double]("Tips") * 20).toDouble / maxTip) + 4)

      val x2 = zippedPlottingData.map(_._1.getAs[Double]("Trip Miles")).filter(_ < 60)
      val y2 = zippedPlottingData.map(_._1.getAs[Double]("prediction"))
      val color2 = RedARGB
      val size2 = zippedPlottingData.map(c => ((c._1.getAs[Double]("Tips") * 20).toDouble / maxTip) + 4)

      val plot = Plot.scatterPlotGrid(Seq(Seq((x2, y2, color2,size2)), Seq((x1, y1,color1, size1))), "Prediction V Acual of Payment Type", "Trip Miles", "Payment Type")
      SwingRenderer(plot, 800, 800, true)
      
    }
    
     println(cluster)
  }
}
package chicagotransportation

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import swiftvis2.plotting.Plot
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.SwingRenderer
import swiftvis2.plotting.ColorGradient

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
        .csv("/data/BigData/students/espradli/taxiData.csv").limit(1000)

      lazy val randomSample = taxiData.randomSplit(Array(.05, .95))(0)
      
      randomSample.write
        //.format("com.databricks.spark.csv").option("header", "true").save("/users/espradli/BigData/taxiDataRandom.csv")
        //.option("header", "true")
        .csv("/data/BigData/students/espradli/taxiDataRandom.csv")
    }
    

  }
}
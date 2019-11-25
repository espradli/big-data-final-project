package chicagotransportation

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import swiftvis2.plotting.Plot
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.SwingRenderer
import swiftvis2.plotting.ColorGradient

object ChicagoTransportation{
  def main(args: Array[String]): Unit ={
    val spark = SparkSession.builder().master("local[*]").appName("SparkSQL2").getOrCreate()
    import spark.implicits._
    
    spark.sparkContext.setLogLevel("WARN")
    
    lazy val taxiData = spark.read.option("inferSchema", true).
      option("header", "true").
      csv("/data/BigData/students/espradli/taxiData.csv")

    val randomSample = taxiData.randomSplit(Array(.05, .995))(0).write.csv("/data/BigData/students/espradli/taxiDataRandom.csv")
  }
}
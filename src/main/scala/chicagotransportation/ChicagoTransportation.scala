package chicagotransportation

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.clustering.KMeans
import swiftvis2.plotting.Plot
import swiftvis2.plotting._
import swiftvis2.plotting.renderer.SwingRenderer
import swiftvis2.plotting.ColorGradient

object ChicagoTransportation{
  def main(args: Array[String]): Unit ={
    val spark = SparkSession.builder().master("local[*]").appName("SparkSQL2").getOrCreate()
    import spark.implicits._
  
    spark.sparkContext.setLogLevel("WARN")

  }
}
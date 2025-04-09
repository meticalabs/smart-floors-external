import pytest
from pyspark import SparkConf
from pyspark.sql import SparkSession


@pytest.fixture
def spark(request):
    # https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr
    import os
    from sys import platform
    dir_path = os.path.dirname(request.module.__file__)

    if platform == "darwin":
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    spark_conf = (
        SparkConf()
        .set("spark.driver.host", "127.0.0.1")
        .set("spark.driver.port", "7077")
        .set("spark.sql.execution.arrow.pyspark.enabled", "true")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .set("spark.sql.catalog.dev", "org.apache.iceberg.spark.SparkCatalog")
        .set("spark.sql.catalog.dev.type", "hadoop")
        .set("spark.sql.catalog.dev.warehouse", f"{dir_path}/../target/warehouse")
        .set("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.7.0")
        .set("spark.sql.shuffle.partitions", "2")
        .set("spark.default.parallelism", "2")
    )
    spark = SparkSession.builder.config(conf=spark_conf).master("local[*]").appName("Local Test Runner").getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    yield spark
    spark.stop()
    if platform == "darwin":
        del os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"]

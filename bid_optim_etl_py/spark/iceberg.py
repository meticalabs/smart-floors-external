import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import boto3
from pyspark.sql import DataFrame, DataFrameWriterV2, SparkSession

logger = logging.getLogger(__name__)


@dataclass
class TableConfig:
    format: str
    database_name: str
    table_name: str

    def db_table_name(self, catalog_name: str = None) -> str:
        return (f"{catalog_name}.{self.database_name}.{self.table_name}"
                if catalog_name else f"{self.database_name}.{self.table_name}")


@dataclass(kw_only=True)
class Maintenance:
    spark: SparkSession
    iceberg_catalog: str

    def __post_init__(self):
        log4jLogger = self.spark.sparkContext._jvm.org.apache.log4j
        self._log = log4jLogger.LogManager.getLogger(__name__)

    def remove_orphan_files(self, db_table_name: str):
        self._log.info(f"Removing orphan files for table {db_table_name}")
        df = self.spark.sql(f"CALL {self.iceberg_catalog}.system.remove_orphan_files(table => '{db_table_name}')")
        df.show(truncate=False)
        self._log.info(f"Orphan files removed for table {db_table_name}")

    def rewrite_data_files(self, db_table_name: str, where_clause: str = None):
        self._log.info(f"Rewriting data files for table {db_table_name}")
        args = {"table": db_table_name, "where": where_clause}
        args_str = ", ".join([f"{k}=>\"{v}\"" for k, v in args.items() if v])
        df = self.spark.sql(f"CALL {self.iceberg_catalog}.system.rewrite_data_files({args_str})")
        df.show(truncate=False)
        self._log.info(f"Data files rewritten for table {db_table_name}")

    def expire_snapshots(self, db_table_name: str, expire_older_than: datetime = None):
        if expire_older_than is None:
            expire_older_than = datetime.now() - timedelta(days=5)
        self._log.info(f"Expiring snapshots for table {db_table_name} older than {expire_older_than}")
        df = self.spark.sql(
            f"CALL {self.iceberg_catalog}.system.expire_snapshots(table => '{db_table_name}', "
            f"older_than => TIMESTAMP '{expire_older_than.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}', "
            f"stream_results => true)")
        df.show(truncate=False)
        self._log.info(f"Snapshots expired for table {db_table_name}")

    def rewrite_manifest_files(self, db_table_name: str):
        self._log.info(f"Rewriting manifest files for table {db_table_name}")
        df = self.spark.sql(f"CALL {self.iceberg_catalog}.system.rewrite_manifests(table => '{db_table_name}')")
        df.show(truncate=False)
        self._log.info(f"Manifest files rewritten for table {db_table_name}")


@dataclass(kw_only=True)
class IcebergIO:
    spark: SparkSession
    region_name: str
    iceberg_catalog: str

    def __post_init__(self):
        log4jLogger = self.spark.sparkContext._jvm.org.apache.log4j
        self._log = log4jLogger.LogManager.getLogger(__name__)

    def configure_spark_writer(self, df: DataFrame, db_table_name: str,
                               options: dict[str, str] = None) -> DataFrameWriterV2:
        df_writer = (
            df.writeTo(db_table_name)
            .tableProperty("write.spark.fanout.enabled", "true")
            .option("mergeSchema", "true")
            .option("check-ordering", "false")
            .option("check-nullability", "false")
        )

        if options and len(options) > 0:
            df_writer = df_writer.options(**options)

        return df_writer

    def create_iceberg_table_if_not_exists(self, df: DataFrame, table_config: TableConfig, partition_by: [str] = None,
                                           table_properties: dict = None):
        db_table_name = table_config.db_table_name(self.iceberg_catalog)
        warehouse_location = self.spark.conf.get(f"spark.sql.catalog.{self.iceberg_catalog}.warehouse")
        if not warehouse_location:
            raise ValueError(
                f"Table location not found for catalog {self.iceberg_catalog}. "
                f"Please check the config `spark.sql.catalog.{self.iceberg_catalog}.warehouse` in spark submit.")

        table_location = f"{warehouse_location.rstrip('/')}/{table_config.database_name}/{table_config.table_name}"

        if not self.spark.catalog.tableExists(db_table_name):
            self._log.info(f"Creating table {db_table_name} at location {table_location}")
            table_properties = table_properties or {}
            table_properties.update({"write.spark.accept-any-schema": "true", "write.wap.enabled": "true"})
            df_writer = df.writeTo(db_table_name).tableProperty("location", table_location).using("iceberg")
            if partition_by:
                df_writer = df_writer.partitionedBy(*[df[col] for col in partition_by])
            df_writer.create()
            table_property_alter = ",".join([f"'{k}'='{v}'" for k, v in table_properties.items()])
            self.spark.sql(
                f"ALTER TABLE {db_table_name} SET TBLPROPERTIES ({table_property_alter})").show()
            self._log.info(f"Table {db_table_name} created at location {table_location} successfully")
        else:
            self._log.info(f"Table {db_table_name} already exists")

    def write(self, spark_df: DataFrame, table_config: TableConfig, partition_by: [str] = None,
              create_database: bool = False, overwrite_partitions: bool = True,
              write_options: dict = None):
        if create_database:
            self.create_glue_database_if_not_exists(name=table_config.database_name, region_name=self.region_name, )

        sink_table_name_with_catalog = table_config.db_table_name(self.iceberg_catalog)
        self.create_iceberg_table_if_not_exists(df=spark_df,
                                                table_config=table_config,
                                                partition_by=partition_by)

        if spark_df.isEmpty():
            self._log.info(f"Dataframe is empty. Skipping write to {sink_table_name_with_catalog}")
        else:
            spark_writer = self.configure_spark_writer(df=spark_df,
                                                       db_table_name=sink_table_name_with_catalog,
                                                       options=write_options)
            if partition_by:
                spark_writer = spark_writer.partitionedBy(*[spark_df[col] for col in partition_by])
            if overwrite_partitions:
                spark_writer.overwritePartitions()
            else:
                spark_writer.append()

    def create_glue_database_if_not_exists(self, region_name: str, name: str, description: str = ""):
        """
        Creates a Glue database if it does not exist
        :param name: Database name
        :param description: Database description
        :return: None
        """
        glue_client = boto3.client(service_name="glue", region_name=region_name)
        try:
            glue_client.get_database(Name=name)
        except glue_client.exceptions.EntityNotFoundException:
            try:
                glue_client.create_database(DatabaseInput={"Name": name, "Description": description})
            except glue_client.exceptions.AlreadyExistsException:
                logging.info(f"Database {name} already exists")

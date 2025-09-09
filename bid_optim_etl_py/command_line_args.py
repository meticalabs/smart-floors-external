import argparse
from abc import ABC, abstractmethod


class CommandLineParser(ABC):
    @staticmethod
    @abstractmethod
    def parser():
        pass

    def parse_args(self, args: [str]):
        return self.parser().parse_args(args=args)


class ApplovinETLConfigParser(CommandLineParser):
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser(description="Run the applovin bid floor training")
        parser.add_argument("--customerId", type=int, help="Customer ID")
        parser.add_argument("--appId", type=int, help="App ID")
        parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
        parser.add_argument(
            "--region", required=True, type=str, help="Region to load config (str). Ex: us-east-1, eu-west-1"
        )
        parser.add_argument("--s3DataBucket", help="S3 bucket name for all events and bid floor training data")
        parser.add_argument(
            "--icebergCatalog", required=True, type=str, help="Iceberg catalog name (str). Ex: iceberg_catalog."
        )
        return parser


class ApplovinModelPublisherArgsParser(CommandLineParser):
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser(description="Run the applovin bid floor model publisher")
        parser.add_argument("--region", help="AWS region where the resources are located", required=True)
        parser.add_argument("--customerId", type=int, help="Customer ID")
        parser.add_argument("--appId", type=int, help="App ID")
        parser.add_argument("--modelIds", nargs="+", type=str, help="Model IDs")
        parser.add_argument("--strategyName", nargs="+", type=str, help="the strategy names")
        parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
        parser.add_argument("--s3ModelArtifactBucket", help="S3 bucket name for model artifact")
        parser.add_argument("--bidFloorVersion", help="Bid floor version")
        parser.add_argument("--allocatorServiceUri", help="Allocator service URI")
        return parser
    

class ApplovinModelTrainingArgsParser(CommandLineParser):
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser(description="Run the applovin bid floor model training")
        parser.add_argument("--region", type=str, help="AWS region name, e.g., us-east-1", required=True)
        parser.add_argument("--customerId", type=int, help="Customer ID")
        parser.add_argument("--appId", type=int, help="App ID")
        parser.add_argument("--modelId", type=str, help="Model ID")
        parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
        parser.add_argument("--icebergTrainDataTable", help="Iceberg db table name for training data")
        parser.add_argument("--s3ModelArtifactBucket", help="S3 bucket name for model artifact")

        return parser


class StrategyTrainingArgsParser(CommandLineParser):
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser(description="Run the strategy training")
        parser.add_argument("--region", type=str, help="AWS region name, e.g., us-east-1", required=True)
        parser.add_argument("--customerId", type=int, help="Customer ID")
        parser.add_argument("--appId", type=int, help="App ID")
        parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
        parser.add_argument("--modelId", type=str, help="Model ID")
        parser.add_argument("--strategyName", type=str, help="strategy name")
        parser.add_argument("--s3ModelArtifactBucket", help="S3 bucket name for model artifact")
        return parser

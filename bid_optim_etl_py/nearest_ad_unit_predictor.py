import logging
import dataclasses

import os
from bid_optim_etl_py.applovin_train_runner import init_ray_cluster
import joblib
import datetime
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, ConfigDict, SkipValidation
from bid_optim_etl_py.cfg_parser import ConfigFile
from bid_optim_etl_py.utils.management_api import BidFloorManagementAPI, HttpClient
from bid_optim_etl_py.command_line_args import StrategyTrainingArgsParser
from etl_py_commons.job_initialiser import Initialisation
import sys
import boto3

# --- S3ModelArtifactInfo and upload_model_file_to_s3 moved here to avoid circular import ---
from dataclasses import dataclass


@dataclass
class S3ModelArtifactInfo:
    bucket: str
    key: str
    file_name: str
    file_name_wo_ext: str


def upload_model_file_to_s3(local_model_base_path, model_artifact_path: S3ModelArtifactInfo):
    boto3_client = boto3.client("s3")
    boto3_client.upload_file(
        os.path.join(local_model_base_path, model_artifact_path.file_name),
        model_artifact_path.bucket,
        model_artifact_path.key,
    )


class NearestAdUnitPredictor(BaseModel):
    """
    This model selects the closest ad unit to the price of the previous aggregated units
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    LOW_MULTIPLIER: float = 0
    HIGH_MULTIPLIER: float = 1.5 * 1000 # Convert to CPM
    rng_exploration: SkipValidation[np.random.Generator] = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )
    rng_shuffle: SkipValidation[np.random.Generator] = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )

    def sort_by_name_postfix_desc(self, assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts a list of assignments in descending order based on the numerical postfix
        in their "name" field. The postfix is determined by splitting the "name" field
        on underscore ('_') and converting the last part into an integer. The sorting
        order prioritizes assignments with higher numerical postfix values.

        :param assignments: A list of dictionaries, each representing an assignment.
        :return: A sorted list of assignments in descending order based on the numerical postfix.
        """
        return sorted(assignments, key=lambda x: int(x["name"].split("_")[-1]), reverse=True)

    def add_hardcoded_contexts(
        self,
        context: pd.Series,
        highest_bid_floor_value: Optional[float],
        medium_bid_floor_value: Optional[float],
    ) -> pd.Series:
        """
        Adds hardcoded context values to the provided context series.
        :param context: The context series to be modified.
        :param highest_bid_floor_value: The highest bid floor value to add to context.
        :param medium_bid_floor_value: The medium bid floor value to add to context.
        :return: The modified context series with hardcoded values added.
        """
        nw = datetime.datetime.now(datetime.timezone.utc)
        context["assignmentDayOfWeek"] = nw.weekday()  # 0-6, Monday-Sunday
        context["assignmentHourOfDay"] = nw.hour
        context["highestBidFloorValue"] = highest_bid_floor_value
        context["mediumBidFloorValue"] = medium_bid_floor_value
        return context

    def split_based_on_name(self, ad_unit_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Splits the ad unit list into two parts: one with the lowest bid floor and the other with the rest.
        :param ad_unit_list: List of ad units to be split.
        :return: Tuple containing the lowest bid floor and the rest of the ad units.
        """
        sorted_by_ad_unit_name = self.sort_by_name_postfix_desc(ad_unit_list)
        return sorted_by_ad_unit_name[:-1], sorted_by_ad_unit_name[-1:][0]

    def form_response(
        self,
        assignments: List[Dict[str, Any]],
        lowest_bid_floor: Dict[str, Any],
        propensity: float,
    ) -> Dict[str, Any]:
        """
        Forms the response dictionary with the predicted reward and other details.
        :param assignments: List of assignments.
        :param lowest_bid_floor: The lowest bid floor value.
        :param propensity: The propensity value.
        :return: Response dictionary.
        """
        assignments = assignments + [lowest_bid_floor]

        cpm_floor_ad_unit_ids = list(map(lambda x: x["id"], assignments))
        cpm_floor_values = list(map(lambda x: x["bidFloor"], assignments))

        response = {
            "cpmFloorAdUnitIds": cpm_floor_ad_unit_ids,
            "cpmFloorValues": cpm_floor_values,
            "propensity": propensity,
        }
        return response

    def predict(
        self, context: pd.Series, floors: List[pd.Series | Dict[str, Any]], max_ad_units: Optional[int] = None
    ) -> Dict[str, Any]:
        floors = [floor.to_dict() if isinstance(floor, pd.Series) else floor for floor in floors]
        
        if not floors:
            raise ValueError("No ad units provided")
        lowest_bid_floor = self.sort_by_name_postfix_desc(floors)[-1]
        if max_ad_units == 1:
            return self.form_response(
                [],
                lowest_bid_floor,
                1.0,
            )
        elif max_ad_units is None or max_ad_units >= 3:
            raise ValueError("NearestAdUnit model only supports max_ad_units of 1 or 2")
        elif max_ad_units == 2:
            # Remove the lowest bid floor from candidates
            candidates = [f for f in floors if f != lowest_bid_floor]
            if not candidates:
                # Only one ad unit, fallback to lowest
                return self.form_response([], lowest_bid_floor, 1.0)
            target = context.get("user.avgInterRevenueLast72Hours", 0) * self.HIGH_MULTIPLIER
            # Find the ad unit whose bidFloor * HIGH_MULTIPLIER is closest to target
            best = min(candidates, key=lambda x: abs(x["bidFloor"] - target))
            return self.form_response([best], lowest_bid_floor, 1.0)


def save_nearest_ad_unit_predictor_model_to_s3(nearest_ad_unit_predictor, app_id, model_artifact_path):
    """
    Save the NearestAdUnitPredictor model to S3.
    """
    logging.info(f"Saving NearestAdUnitPredictor model to S3: {model_artifact_path.bucket}/{model_artifact_path.key}")
    local_model_base_path = os.path.join("/tmp", str(app_id))
    if not os.path.exists(local_model_base_path):
        os.makedirs(local_model_base_path)
    joblib.dump(nearest_ad_unit_predictor, os.path.join(local_model_base_path, model_artifact_path.file_name))
    upload_model_file_to_s3(local_model_base_path, model_artifact_path)


def save_nearest_ad_unit_predictor_object(nearest_ad_unit_predictor, args):
    predictor_file_name = f"{args.customerId}_{args.appId}_{args.modelId}_{args.strategyName}"
    predictor_final_name_with_ext = f"{predictor_file_name}.joblib"
    logging.info(
        f"Saving NearestAdUnitPredictor predictor object to S3: "
        f"  bid_floor_models/{args.date}/{predictor_final_name_with_ext}"
    )

    save_nearest_ad_unit_predictor_model_to_s3(
        nearest_ad_unit_predictor,
        args.appId,
        S3ModelArtifactInfo(
            bucket=args.s3ModelArtifactBucket,
            key=f"bid_floor_models/{args.date}/{predictor_final_name_with_ext}",
            file_name=predictor_final_name_with_ext,
            file_name_wo_ext=predictor_file_name,
        ),
    )


def run():
    argvs = sys.argv[1:] if len(sys.argv) > 1 else []
    parsed_args_obj = Initialisation.parse_args(args=argvs, parser_obj=StrategyTrainingArgsParser())
    cmd_line_args = parsed_args_obj.parsed_args
    config_file = parsed_args_obj.read_config(
        confs_dir_path=os.path.join(os.path.dirname(__file__), "confs"), config_clazz_type=ConfigFile
    )

    bid_floor_management_api = BidFloorManagementAPI(http_client=HttpClient(base_url=config_file.managementApiBaseUrl))
    etl_config = bid_floor_management_api.fetch_etl_config(cmd_line_args.appId)
    model_config = bid_floor_management_api.fetch_model_config(cmd_line_args.appId, cmd_line_args.modelId)

    logging.info(f"ETL Config: {etl_config}")
    logging.info(f"Model Config: {model_config}")

    init_ray_cluster()
    logging.info("Creating model as it is requested in the args")
    save_nearest_ad_unit_predictor_object(
        nearest_ad_unit_predictor=NearestAdUnitPredictor(),
        args=cmd_line_args,
    )


if __name__ == "__main__":
    run()

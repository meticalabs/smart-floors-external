import json
import os.path

import joblib
import pandas as pd


class ContextualBanditModelHandler(object):

    def __init__(self):
        self.initialized = False
        self.models = None

    def initialize(self, context):
        self.initialized = True
        model_dir = context.system_properties.get("model_dir")
        model_tar_extracted_dir = next(x for x in os.listdir(model_dir) if not x.startswith("."))
        predict_joblib_path = os.path.join(model_dir, f"{model_tar_extracted_dir}/predictor.joblib")
        print(f"Reading the files from {predict_joblib_path}")
        self.models = joblib.load(predict_joblib_path)
        print(f"Totals present, [Models : {len(self.models)}]")

    def preprocess_single_request(self, body):
        model_id = body["modelId"]
        context_series = pd.Series(body["context"])
        return {
            "model_id": model_id,
            "user_id": body["userId"],
            "context_series": context_series,
            "ad_units": body["adUnits"]
        }

    def preprocess(self, request):
        input_context_vectors: [dict] = []
        is_batch = False
        for single_req in request:
            single_req_body = json.loads(single_req.get("body").decode("utf-8"))
            if "users" not in single_req_body:
                input_context_vectors.append(self.preprocess_single_request(single_req_body))
            else:
                is_batch = True
                for body in single_req_body["users"]:
                    input_context_vectors.append(self.preprocess_single_request(body))
        return input_context_vectors, is_batch

    def fetch_featuriser(self, experiment_id):
        if experiment_id not in self.featurisers:
            raise ValueError(f"Experiment {experiment_id} not found in featurisers")
        return self.featurisers[experiment_id]

    def fetch_model(self, model_id):
        if model_id not in self.models:
            raise ValueError(f"Model ID {model_id} not found in contextual models")
        return self.models[model_id]

    def handle(self, data, context):
        preprocessed_request, is_batch = self.preprocess(data)
        output = []
        for context_data in preprocessed_request:
            model = self.fetch_model(context_data["model_id"])
            bid_floor_response = model.predict(context_data["context_vector_series"], context_data["ad_units"])
            bid_floor_response.update(
                {
                    "userId": context_data["user_id"],
                    "modelId": context_data["model_id"],
                }
            )
            output.append(bid_floor_response)

        return [{"allocations": output}] if is_batch else output


_service = ContextualBanditModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

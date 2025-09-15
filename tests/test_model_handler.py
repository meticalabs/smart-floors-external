import os
import joblib
import json
from bid_optim_etl_py.nearest_ad_unit_predictor import NearestAdUnitPredictor
import importlib.util
import sys

# Dynamically import model_handler from sagemaker-inference/model_handler.py
spec = importlib.util.spec_from_file_location(
    "model_handler",
    os.path.join(os.path.dirname(__file__), "../sagemaker-inference/model_handler.py")
)
model_handler = importlib.util.module_from_spec(spec)
sys.modules["model_handler"] = model_handler
spec.loader.exec_module(model_handler)

test_event={
        "users": [
            {
            "context": {
                "user.minRevenueLast24Hours": 0.001061701551189245,
                "user.avgRevenueLast24Hours": 0.001061701551189245,
                "user.avgInterRevenueasdLast72Hours": 0.001061701551189245,
                "user.mostRecentAdRevenue": 0.001061701551189245,
                "user.platform": "android",
                "user.adformat": "inter",
                "user.country": "UK",
                "user.languageCode": "en-US"
            },
            "adUnits": [
                {
                "id": "8c63ff91a4a47584",
                "name": "METICA_AD_UNIT_1",
                "bidFloor": 0.1
                },
                {
                "id": "e6204ff1d59d2894",
                "name": "METICA_AD_UNIT_2",
                "bidFloor": 0.5
                },
                {
                "id": "37f4c63d4e3723e6",
                "name": "METICA_AD_UNIT_3",
                "bidFloor": 1.0
                },
                {
                "id": "bf648cbdf2a10f2c",
                "name": "METICA_AD_UNIT_4",
                "bidFloor": 5.0
                },
                {
                "id": "bde460c060b9a272",
                "name": "METICA_AD_UNIT_5",
                "bidFloor": 10.0
                }
            ],
            "userId": "2779d449b8f55162",
            "modelId": "android_inter",
            "reference": "15118",
            "maxAdUnits": 2,
            "assignmentStickinessInSeconds": 7200
            }
        ]
    }

def make_fake_context(model_dir):
    class FakeContext:
        system_properties = {"model_dir": model_dir}
    return FakeContext()

def test_model_handler_with_nearest_ad_unit_predictor(tmp_path):
    # Prepare a fake model dict structure as expected by model_handler
    predictor = NearestAdUnitPredictor()
    models = {"15118": {"android_inter": predictor}}
    # Save the model dict as joblib
    model_dir = tmp_path / "model"
    os.makedirs(model_dir)
    extracted_dir = model_dir / "extracted"
    os.makedirs(extracted_dir)
    joblib.dump(models, extracted_dir / "predictor.joblib")
    # Patch os.listdir to simulate SageMaker extraction
    orig_listdir = os.listdir
    os.listdir = lambda d: ["extracted"]
    try:
        context = make_fake_context(str(model_dir))
        handler = model_handler.ContextualBanditModelHandler()
        handler.initialize(context)
        # Prepare request
        request = [{"body": bytes(json.dumps(test_event), "utf-8") }]
        result = handler.handle(request, context)
        allocations = result[0]["allocations"]
        assert isinstance(allocations, list)
        assert allocations[0]["userId"] == "2779d449b8f55162"
        assert allocations[0]["modelId"] == "android_inter"
        assert allocations[0]["reference"] == "15118"
        assert allocations[0]["cpmFloorValues"] == [1.0, 0.1]
        assert allocations[0]["cpmFloorAdUnitIds"] == ["37f4c63d4e3723e6", "8c63ff91a4a47584"]
        assert "cpmFloorAdUnitIds" in allocations[0]
    finally:
        os.listdir = orig_listdir


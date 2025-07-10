
import os
import tarfile
import joblib
import argparse
import tempfile
import shutil
from bid_optim_etl_py.applovin_train_runner import Predictor, ValueReplacer, Features

def create_empty_model_artifact(output_tar_file):
    """
    Creates an empty model artifact and saves it to the specified tarball path.
    """
    # Create an empty Predictor object
    empty_predictor = Predictor(
        epsilon=0.1,
        clf=None,
        value_replacer=ValueReplacer(valid_values={}, default_value="other"),
        features=Features([]),
    )

    # Create the dictionary to be saved in the joblib file
    model_dict = {"default_bid_floor": empty_predictor}

    # Create a temporary directory to stage the artifact
    staging_dir = tempfile.mkdtemp()

    try:
        joblib_file = os.path.join(staging_dir, "predictor.joblib")
        joblib.dump(model_dict, joblib_file)

        # Create the tarball
        with tarfile.open(output_tar_file, "w:gz") as tar:
            tar.add(joblib_file, arcname="predictor.joblib")

        print(f"Empty model artifact created at: {output_tar_file}")

    finally:
        # Clean up the temporary directory
        shutil.rmtree(staging_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an empty model artifact.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output tar.gz file.")
    args = parser.parse_args()

    create_empty_model_artifact(args.output_file)

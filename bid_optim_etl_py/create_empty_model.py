import argparse
import os
import shutil
import tarfile
import tempfile

import joblib

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
    model_dict = {
        "android_inter": empty_predictor,
        "android_reward": empty_predictor,
        "ios_inter": empty_predictor,
        "ios_reward": empty_predictor,
    }

    # Create a temporary directory to stage the artifact
    staging_dir = os.path.join(tempfile.mkdtemp(), "empty_model_artifact")

    try:
        os.makedirs(staging_dir, exist_ok=True)

        joblib_file = os.path.join(staging_dir, "predictor.joblib")
        joblib.dump(model_dict, joblib_file)
        tar_file_name = os.path.basename(output_tar_file)

        # Create the tarball
        with tarfile.open(os.path.join(staging_dir, tar_file_name), "w:gz") as tar:
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

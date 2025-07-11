import argparse
import os
import shutil
import tarfile

import joblib

from bid_optim_etl_py.applovin_train_runner import Predictor, ValueReplacer, Features


def create_tar_archive(source_dir, output_filename):
    print(f"Creating tar archive from {source_dir} to {output_filename}")
    with tarfile.open(os.path.join(os.path.dirname(source_dir), output_filename), "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


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
        "default_bid_floor": {
            "android_inter": empty_predictor,
            "android_reward": empty_predictor,
            "ios_inter": empty_predictor,
            "ios_reward": empty_predictor,
        }
    }

    # Create a temporary directory to stage the artifact
    tar_file_name = os.path.basename(output_tar_file).replace(".tar.gz", "")
    tmp_dir = os.path.dirname(output_tar_file)
    staging_dir = os.path.join(tmp_dir, tar_file_name)
    try:
        os.makedirs(staging_dir, exist_ok=True)

        joblib_file = os.path.join(staging_dir, "predictor.joblib")
        joblib.dump(model_dict, joblib_file)

        create_tar_archive(staging_dir, os.path.basename(output_tar_file))

        print(f"Empty model artifact created at: {output_tar_file}")

    finally:
        # Clean up the temporary directory
        shutil.rmtree(staging_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an empty model artifact.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output tar.gz file.")
    args = parser.parse_args()

    create_empty_model_artifact(args.output_file)

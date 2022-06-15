import traceback
import argparse
from pathlib import Path
import os

from dlacc.optimum import Optimum
from dlacc.utils import (
    convert2onnx,
    JSONConfig,
    JSONOutput,
)
from dlacc.metadata import platformType, output_prefix, input_prefix
from app.helpers import publish_message, download_file_from_gcp, upload_blobs_from_directory

def run(args: dict) -> None:
    config_name = "config.json"
    model_name = "model.onnx"
    try:
        if args.env == platformType.GOOGLESTORAGE:
            download_file_from_gcp(args.input_bucket, f"job_id={str(args.job_id)}/{config_name}", input_prefix, config_name)
            download_file_from_gcp(args.input_bucket, f"job_id={str(args.job_id)}/{model_name}", input_prefix, model_name)
        
        config = JSONConfig(config_name, 0)
        onnx_model = convert2onnx(
            0,
            f"{input_prefix}/{model_name}",
            config["model_type"],
            input_shape=config["model_config"]["input_shape"],
            input_dtype=config["model_config"]["input_dtype"],
        )

        out_json = JSONOutput(config)
        out_json["status"] = 1

        try:
            optimum = Optimum(out_json["model_name"])
            optimum.run(onnx_model, out_json)
        except Exception as e:
            traceback.print_exc()
            out_json["error_info"] = str(e)
            out_json["status"] = -1

        if out_json["status"] != -1:
            out_json["status"] = 4
        out_json.save(output_prefix + "/output_json.json")

        if config["need_benchmark"]:
            optimum.ansor_engine.evaluate()

        if args.env == platformType.GOOGLESTORAGE:
            upload_blobs_from_directory(output_prefix, args.output_bucket, "job_id=%s" % args.job_id)
            publish_message(args.project_id, args.topic_id, "OK", job_id = args.job_id, job_status = str(out_json["status"]))

    except Exception as e:
        publish_message(args.project_id, args.topic_id, str(e), job_id = args.job_id, job_status = "-1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=int,
        help="The path of config file in json format.",
        required=True,
    )

    parser.add_argument(
        "--job_id",
        type=int,
        help="The path of config file in json format.",
        required=True,
    )

    parser.add_argument(
        "--input_bucket",
        type=str,
        help="The path of config file in json format.",
        required=True,
    )
 
    parser.add_argument(
        "--output_bucket",
        type=str,
        help="The path of config file in json format.",
        required=True,
    )

    parser.add_argument(
        "--topic_id",
        type=str,
        help="The path of config file in json format.",
        required=True,
    )

    parser.add_argument(
        "--project_id",
        type=str,
        help="The path of config file in json format.",
        required=True,
    )

    args = parser.parse_args()

    run(args)

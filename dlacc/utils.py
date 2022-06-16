from pathlib import Path
import os
import json
import re
import glob
import numpy as np
import onnx

from base_class import BaseClass
from metadata import ModelType

def contruct_dummy_input(input_shape, input_dtype, tensor_type, device="cuda"):
    import numpy as np

    if tensor_type == "pt":
        import torch

        dummy_input = tuple(
            [
                torch.randn(*v).type(
                    {
                        "int32": torch.int32,
                        "int64": torch.int64,
                        "float32": torch.float32,
                        "float64": torch.float64,
                    }[input_dtype[k]]
                )
                for k, v in input_shape.items()
            ]
        )
    elif tensor_type == "tf":
        import tensorflow as tf

        dummy_input = [
            tf.TensorSpec(
                v,
                {
                    "int32": tf.int32,
                    "int64": tf.int64,
                    "float32": tf.float32,
                    "float64": tf.float64,
                }[input_dtype[k]],
                name="x",
            )
            for k, v in input_shape.items()
        ]
    else:
        dummy_input = dict(
            [
                (k, np.random.rand(*v).astype(input_dtype[k]))
                for k, v in input_shape.items()
            ]
        )
    return dummy_input


def get_onnx_model(model_path, model_type):
    if model_type == int(ModelType.ONNX):
        onnx_model = onnx.load(model_path)
    else:
	raise NotImplementedError

    return onnx_model


class JSONConfig(BaseClass):
    def __init__(self, json_path) -> None:
        self.load(json_path)

    def load(self, json_path):
        with open(json_path) as json_file:
            self.meta = json.load(json_file)

    def __getitem__(self, key):
        return self.meta[key]


class JSONOutput(BaseClass):
    def __init__(self, json_config: JSONConfig):
        self.meta = json_config.meta

    def save(self, file_path):
        with open(file_path, "w") as outfile:
            json.dump(self.meta, outfile)

    def __getitem__(self, key):
        return self.meta[key]

    def __setitem__(self, key, value):
        self.meta[key] = value

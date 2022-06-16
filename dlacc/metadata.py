import enum


class platformType(enum.IntEnum):
    LOCAL = 0
    GOOGLESTORAGE = 1
    AWSSTORAGE = 2


class ModelType(enum.IntEnum):
    PT = 0
    TF = 1
    ONNX = 2
    KERAS = 3


input_prefix = "../inputs"
output_prefix = "../outputs"

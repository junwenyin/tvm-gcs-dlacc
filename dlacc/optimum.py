"""! @brief Defines the Optimum class."""
##
# @file optimum.py
#
# @brief Defines the Optimum class.
#
# @section author_sensors Author(s)
# - Created by Gnomondigital on 02/06/2022.
# - Modified by Gnomondigital on 02/06/2022.
#
# Copyright (c) 2022 Gnomondigital.  All rights reserved.

from tvm.contrib import graph_executor
import tvm
from ansor_engine import AnsorEngine
from base_class import BaseClass
import onnx


class GraphModuleWrapper:
    """A wrapper class for graph module. It is the final object for prediction."""    
    def __init__(self, module: tvm.contrib.graph_executor.GraphModule):
        self.module = module

    def __call__(self, inputs_dict: dict) -> dict:
        """Run prediction.

        Parameters
        ----------
        inputs_dict : dict
            The input dictionnary. The keys are input names, values are their numeric values.

        Returns
        -------
        dict
            Output dictionnary with output names and values.
        """
        self.module.set_input(**inputs_dict)
        self.module.run()
        num_outputs = self.module.get_num_outputs()
        tvm_outputs = {}
        for i in range(num_outputs):
            output_name = "output_{}".format(i)
            tvm_outputs[output_name] = self.module.get_output(i).numpy()
        return tvm_outputs

    def predict(self, inputs_dict) -> dict:
        """Run prediction.

        Parameters
        ----------
        inputs_dict : dict
            The input dictionnary. The keys are input names, values are their numeric values.

        Returns
        -------
        dict
            Output dictionnary with output names and values.
        """        
        return self.__call__(inputs_dict)


class Optimum(BaseClass):
    """Optimization entry class.

    Parameters
    ----------
    model_name : str
        The name of model.
    """
    def __init__(self, model_name: str):
        """Initializer.

        Parameters
        ----------
        model_name : str
            The name of model.
        """        
        self.model_name = model_name

    def run(self, onnx_model, config: dict):
        """Run tuning process.

        Parameters
        ----------
        onnx_model : onnx.onnx_ml_pb2.ModelProto
                Onnx model object.
        config: dict
            The output json dictionnary which contains information about runtime.

        Returns
        -------
        None
        """
        return self._run(
            onnx_model,
            config["target"],
            config["tuning_config"]["num_measure_trials"],
            config["tuning_config"]["mode"],
            config,
            log_file=config["tuned_log"],
            input_shape=config["model_config"]["input_shape"],
            input_dtype=config["model_config"]["input_dtype"],
            verbose=config["tuning_config"]["verbose_print"],
        )

    def _run(
        self,
        onnx_model: onnx.onnx_ml_pb2.ModelProto,
        target: str,
        num_measure_trials: int,
        mode: str,
        out_json: dict,
        input_shape: list[int]= None,
        input_dtype: list[str]=None,
        log_file: str = None,
        verbose: int = 0,
    ):
        """Entrypoint for AnsorEngine. See comments in related methodes in AnsorEngine."""    
        if mode == "ansor":
            ae = AnsorEngine(
                self.model_name,
                onnx_model,
                target,
                input_shape,
                input_dtype,
                out_json,
            )
            if log_file != "":
                print(
                    "Historical configuration file %s found, tuning will not be executed."
                    % log_file
                )
                ae.ansor_compile(log_file=log_file)
            else:
                ae.ansor_run_tuning(
                    num_measure_trials=num_measure_trials, verbose=verbose
                )
        elif mode == "autotvm":
            raise NotImplementedError
        self.ansor_engine = ae
        self.onnx_model = onnx_model

    def get_model(self):
        return GraphModuleWrapper(self.ansor_engine.module)

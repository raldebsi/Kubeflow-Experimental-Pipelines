import json
from typing import NamedTuple, Optional

import kfp
import yaml
from kfp import components
from kfp.components import func_to_container_op
from src.pipelines.common_utils import (add_pvolumes_func, get_volume_by_name, setup_volume,
                                        spec_from_file_format)


def create_serve_task(dataset_name, experiment_name, mount_name):

    gen_name_comp = func_to_container_op(generate_inference_service_name)
    gen_name_task = gen_name_comp(dataset_name, experiment_name)
    model_name = gen_name_task.output
    
    infer_service = spec_from_file_format(
        "src/pipelines/yamls/Specs/KServe_HF.yaml",
        modelName=model_name,
        modelNamespace="{{workflow.namespace}}",
        volumeResourceName=mount_name,
        experimentName=experiment_name,
        datasetName=dataset_name
    )

    serve_op = components.load_component_from_file("src/pipelines/yamls/Components/kfserve_launcher.yaml")
    serve_task = serve_op(
        action="apply",
        inferenceservice_yaml=yaml.dump(infer_service),
    )
    
    return serve_task

def generate_inference_service_name(dataset_name, experiment_name) -> str:
    return "{}.{}".format(dataset_name, experiment_name).lower().replace('_', '-')

@kfp.dsl.pipeline(
    name="End to End Hugging Face Topic Classifier",
    description="End to End Topic Classiciation using HuggingFace Framework and CamelBert model"
)
def pipeline(experiment_name: str, volume_name: str, dataset_name: str): 
    volumetrize = setup_volume(volume_name, "/store")

    serve_task = create_serve_task(dataset_name, experiment_name, volume_name)
    serve_task = volumetrize(serve_task)

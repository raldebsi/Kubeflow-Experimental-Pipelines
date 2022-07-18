import json
from typing import NamedTuple, Optional

import kfp
import yaml
from kfp import components
from kfp.components import func_to_container_op
from src.pipelines.common_utils import (add_pvolumes_func, get_volume_by_name, setup_volume,
                                        spec_from_file_format)


def create_serve_task(dataset_name, experiment_name, mount_name):

    # gen_name_comp = func_to_container_op(generate_inference_service_name)
    # gen_name_task = gen_name_comp(dataset_name, experiment_name)
    
    infer_service = spec_from_file_format(
        "src/pipelines/yamls/Specs/KServe_HF.yaml",
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
    
    # return serve_task, service_name
    return serve_task

def generate_inference_service_name(dataset_name, experiment_name) -> str:
    return "{}-{}".format(dataset_name, experiment_name).lower().replace('_', '-')

def create_handler(handler_code: str) -> NamedTuple(
    "Outputs", [
        ("handler_path", str),
        ("requirements_path", str)
    ]
):
    with open("/tmp/handler.py", "w", encoding='utf-8') as f:
        f.write(handler_code)
    
    with open("/tmp/requirements.txt", "w", encoding='utf-8') as f:
        for r in [
            "anltk"
        ]:
            f.write("{}\n".format(r))

    return "/tmp/handler.py", "/tmp/requirements.txt"

def convert_model_to_mar(model_name: str, experiment_name: str, root_path: str) -> str:
    import os
    # This should be a separate function, one to move he files and so on, and one to do the task


def get_mar_required_files(experiment_name: str, dataset_name: str, root_path: str) -> NamedTuple(
    "Outputs", [
        ("model_path", str),
        ("extra_files", str)
    ]
):
    import os
    # storageUri: "pvc://{volumeResourceName}/{experimentName}/outputs/{datasetName}"
    model_folder = os.path.join(root_path, experiment_name, "outputs", dataset_name)
    model = os.path.join(model_folder, "pytorch_model.bin")
    vocab = os.path.join(model_folder, "vocab.txt")
    config = os.path.join(model_folder, "config.json")

    return model, "{},{}".format(vocab, config)

@kfp.dsl.pipeline(
    name="End to End Hugging Face Topic Classifier - Serving",
    description="The Serving part of the E2E HF Topic Classifier - DEBUG ONLY"
)
def pipeline(experiment_name: str, volume_name: str, dataset_name: str, model_version: str = "0.0.1"): 
    volumetrize = setup_volume(volume_name, "/store")

    serve_task, service_name = create_serve_task(dataset_name, experiment_name, volume_name)
    serve_task = volumetrize(serve_task)

    # Convert this to a utility function
    handler_import_path = "src/handlers/topic_class.py"
    handler_code = open(handler_import_path, encoding='utf-8').read()

    handler_op = components.func_to_container_op(create_handler)
    handler_task = handler_op(handler_code)
    handler_task = volumetrize(handler_task) # To allow saving

    get_mar_op = components.func_to_component_op(get_mar_required_files)
    get_mar_task = get_mar_op(experiment_name, dataset_name, "/store")

    model_path = get_mar_task.outputs.model_path
    extra_files = get_mar_task.outputs.extra_files
    model_name = dataset_name

    handler_path = handler_task.outputs.handler_path
    requirements_path = handler_task.outputs.requirements_path

    convertor_op = components.create_component_from_func(
        convert_model_to_mar,
        base_image="python:3.7",
        packages_to_install=['transformers[torch]', 'fastapi[standard]', 'uvicorn[standard]', 'anltk', "torch-model-archiver"],
        command=["torch-model-archiver"],
        args=[ # If this doesn't work then create a component yaml for it
            "--model-name", model_name,
            "--version", model_version,
            "--serialized-file", model_path,
            "--extra-files", extra_files,
            "--handler", handler_path,
            "--requirements-file", requirements_path,
        ]
    )


    # fastapi_deploy_op = kfp.components.create_component_from_func(
	# 	fastapi_deploy,
	# 	packages_to_install=['transformers[torch]', 'fastapi[standard]', 'uvicorn[standard]', 'anltk']
	# )

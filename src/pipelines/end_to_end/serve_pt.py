import json
from typing import NamedTuple, Optional

import kfp
import yaml
from kfp import components
from kfp.components import func_to_container_op
from src.pipelines.common_utils import (add_pvolumes_func, cacheless_task, get_volume_by_name, sanitize_service, setup_volume,
                                        spec_from_file_format)


def create_serve_task(dataset_name: str, experiment_name: str, mount_name: str):
    from uuid import uuid4
    # gen_name_comp = func_to_container_op(generate_inference_service_name)
    # gen_name_task = gen_name_comp(dataset_name, experiment_name)

    sane_service_name = sanitize_service(experiment_name, True)
    
    infer_service = spec_from_file_format(
        "src/pipelines/yamls/Specs/KServe_HF.yaml",
        modelNamespace="{{workflow.namespace}}",
        serviceName=sane_service_name,
        volumeResourceName=mount_name,
        experimentName=experiment_name,
        datasetName=dataset_name,
    )

    serve_op = components.load_component_from_file("src/pipelines/yamls/Components/kserve_launcher.yaml")
    serve_task = serve_op(
        action="apply", 
        inferenceservice_yaml=yaml.dump(infer_service),
        # model_name=sane_service_name,
        # model_uri="pvc://{}/{}".format(mount_name, experiment_name),
        # framework="pytorch",
        # namespace="{{workflow.namespace}}",
        # enable_istio_sidecar=False,
    )
    
    # return serve_task, service_name
    return serve_task


def generate_inference_service_name(dataset_name, experiment_name) -> str:
    return "{}-{}".format(dataset_name, experiment_name).lower().replace('_', '-')

class OHandler(NamedTuple):
    temp_path: str
    handler_path: str
    requirements_path: str

def create_handler(handler_code: str, root_path: str) -> OHandler:
    import os

    temp_path = os.path.join(root_path, "tmp")
    os.makedirs(temp_path, exist_ok=True)
    handler = os.path.join(temp_path, "handler.py")
    requirements = os.path.join(temp_path, "requirements.txt")

    with open(handler, "w", encoding='utf-8') as f:
        f.write(handler_code)
    
    with open(requirements, "w", encoding='utf-8') as f:
        for r in [
            "anltk",
            "torchserve",
            "transformers",
        ]:
            f.write("{}\n".format(r))

    return temp_path, handler, requirements # type: ignore

def move_mar_model(model_name: str, experiment_name: str, root_path: str) -> None: # Can be replaced with text component
    import os
    import shutil

    mar_name = "{}.mar".format(model_name)
    model_store_path = os.path.join(root_path, experiment_name, "model-store")
    mar_model = os.path.join(root_path, "tmp", mar_name)
    
    if os.path.exists(mar_model):
        os.makedirs(model_store_path, exist_ok=True)
        move_path = os.path.join(root_path, model_store_path)
        file_path = os.path.join(move_path, mar_name)
        if os.path.exists(file_path):
            os.remove(file_path)
        shutil.move(mar_model, move_path)
    else:
        raise Exception("Model not found at" + mar_model)        
    
    
class OMarFiles(NamedTuple):
    model_path: str
    extra_files: str

def get_mar_required_files(experiment_name: str, dataset_name: str, root_path: str) -> OMarFiles:
    import os
    # storageUri: "pvc://{volumeResourceName}/{experimentName}/outputs/{datasetName}"
    model_folder = os.path.join(root_path, experiment_name, "outputs", dataset_name)
    model = os.path.join(model_folder, "pytorch_model.bin")
    vocab = os.path.join(model_folder, "vocab.txt")
    config = os.path.join(model_folder, "config.json")

    return model, "{},{}".format(vocab, config) # type: ignore 

@kfp.dsl.pipeline(
    name="End to End Hugging Face Topic Classifier - Serving",
    description="The Serving part of the E2E HF Topic Classifier - DEBUG ONLY"
)
def pipeline(experiment_name: str, volume_name: str, dataset_name: str, model_version: str = "0.0.1"): 
    mount_dir = "/store"
    volumetrize = setup_volume(volume_name, mount_dir)

    # Convert this to a utility function
    handler_import_path = "src/handlers/topic_class.py"
    handler_code = open(handler_import_path, encoding='utf-8').read() 

    handler_op = components.func_to_container_op(create_handler)
    handler_task = handler_op(handler_code, mount_dir)
    handler_task = volumetrize(handler_task) # To allow saving

    get_mar_op = components.func_to_container_op(get_mar_required_files)
    get_mar_task = get_mar_op(experiment_name, dataset_name, mount_dir)

    model_path = get_mar_task.outputs["model_path"]
    extra_files = get_mar_task.outputs["extra_files"]
    model_name = dataset_name

    handler_path = handler_task.outputs["handler_path"]
    requirements_path = handler_task.outputs["requirements_path"]
    out_temp_path = handler_task.outputs["temp_path"]

    mar_convert_op = components.load_component_from_file("src/pipelines/yamls/Components/hf_make_mar_file.yaml")
    mar_convert_task = mar_convert_op(
        model_name = model_name,
        model_version = model_version,
        export_path = out_temp_path,
        model_file = model_path,
        extra_files = extra_files,
        handler_file = handler_path,
        requirements_file = requirements_path, 
    )
    mar_convert_task = volumetrize(mar_convert_task)
    cacheless_task(mar_convert_task)

    move_model_op = components.func_to_container_op(move_mar_model)
    move_model_task = move_model_op(model_name, experiment_name, mount_dir)
    move_model_task = volumetrize(move_model_task)
    move_model_task = move_model_task.after(mar_convert_task)
    cacheless_task(move_model_task)


    serve_task = create_serve_task(dataset_name, experiment_name, volume_name)
    serve_task = volumetrize(serve_task)
    serve_task = serve_task.after(move_model_task)
    cacheless_task(serve_task)

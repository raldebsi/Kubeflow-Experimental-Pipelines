from typing import NamedTuple

import kfp
import yaml
from kfp import components
from kfp.components import func_to_container_op
from src.pipelines.common_utils import (add_pvolumes_func, cacheless_task, generate_random_hex_service, get_volume_by_name, sanitize_service, setup_volume,
                                        spec_from_file_format)


def create_serve_task(dataset_name: str, experiment_name: str, mount_name: str, randomize_service_suffix: bool, use_seed: bool):
    # gen_name_comp = func_to_container_op(generate_inference_service_name)
    # gen_name_task = gen_name_comp(dataset_name, experiment_name)

    sane_service_name = sanitize_service(experiment_name, randomize_service_suffix)
    randomSeed = generate_random_hex_service(use_seed)
    
    infer_service = spec_from_file_format(
        "src/pipelines/yamls/Specs/KServe_HF.yaml",
        modelNamespace="{{workflow.namespace}}",
        serviceName=sane_service_name,
        volumeResourceName=mount_name,
        experimentName=experiment_name,
        datasetName=dataset_name,
        randomSeed=randomSeed, # Prevent reuse of revision
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
    
    return serve_task


def generate_inference_service_name(dataset_name, experiment_name) -> str:
    return "{}-{}".format(dataset_name, experiment_name).lower().replace('_', '-')
 
class OHandler(NamedTuple):
    temp_path: str
    handler_path: str
    requirements_path: str

def create_handler(handler_code: str, experiment_name: str, root_path: str, additional_requirements: list) -> OHandler:
    import os

    temp_path = os.path.join(root_path, experiment_name, "handler")
    os.makedirs(temp_path, exist_ok=True)
    handler = os.path.join(temp_path, "handler.py")
    requirements = os.path.join(temp_path, "requirements.txt")

    print("Temp Path", temp_path)
    print("Handler", handler)
    print("Requirements", requirements)

    with open(handler, "w", encoding='utf-8') as f:
        f.write(handler_code)
    
    with open(requirements, "w", encoding='utf-8') as f:
        for r in additional_requirements:
            f.write("{}\n".format(r))

    return temp_path, handler, requirements # type: ignore

def create_mar_config(experiment_name: str, root_path: str, mar_file: str, model_name: str = "", model_version: str = "1.0", # Pipeline Config
        threads_count: int = 4, job_queue_size: int = 10, install_dependencies: bool = True, is_default: bool = True, # Config Config
        workers_count: int = 1, workers_max: int = 5, batch_size: int = 1, timeout: int = 120, # Model Config
) -> None:
    import os
    import json

    mar_file = mar_file if mar_file.endswith(".mar") else mar_file + ".mar"
    model_name = model_name or mar_file.rsplit(".", 1)[0] # If Empty replace with mar file
    model_name = model_name.replace("-", "").lower()

    experiment_path = os.path.join(root_path, experiment_name)
    print("Experiment Path is", experiment_path)
    model_store = os.path.join(experiment_path, "model-store")
    config_path = os.path.join(experiment_path, "config", "config.properties")

    previous_models = {}

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line[0] == "#":
                    continue
                key, val = line.split("=")
                if key == "model_snapshot":
                    previous_models = json.loads(val.strip())
    if previous_models:
        print("Found Previous Snapshot")
        print(previous_models)
    # mar_files = [ file.rsplit(".", 1)[0]
    #     for file in os.listdir(model_store)
    #     if file.lower().endswith(".mar")
    # ]

    if model_version in previous_models.get("models", {}).get(model_name, {}):
        replace_mode = True
    else:
        replace_mode = False

    model_snapshot = {
        "name": "startup.cfg",
        "modelCount": previous_models.get("modelCount", 0) + (1 if not replace_mode else 0),
        "models": previous_models.get("models", {}) 
    }

    model_dict = model_snapshot["models"].setdefault(model_name, {})
    if is_default:
        print("Setting Default Model to", model_version)
        for model_version_dict in model_snapshot["models"][model_name].values():
            model_version_dict["defaultVersion"] = False # Reset all to false since I am the new default
    else:
        print("Default model unchanged")
    
    model_dict[model_version] = {
        "defaultVersion": is_default,
        "marName": mar_file,
        "minWorkers": workers_count,
        "maxWorkers": workers_max,
        "batchSize": batch_size,
        "maxBatchDelay": 5000,
        "responseTimeout": timeout,
    }

    config_spec = {
        "inference_address": "http://0.0.0.0:8085",
        "management_address": "http://0.0.0.0:8085",
        "metrics_address": "http://0.0.0.0:8082",
        "enable_metrics_api": True,
        "metrics_format": "prometheus",
        "number_of_netty_threads": threads_count,
        "job_queue_size": job_queue_size,
        "model_store": "/mnt/models/model-store",
        "model_snapshot": json.dumps(model_snapshot),
        "install_py_dep_per_model": install_dependencies,
    }

    print("Saving config to", config_path)
    with open(config_path, "w", encoding='utf-8') as f:
        for key, val in config_spec.items():
            if isinstance(val, (set, tuple)):
                val = list(val)
            if isinstance(val, (bool, dict, list)):
                val = json.dumps(val)
            s = "{}={}".format(key, val)
            print(s)
            f.write(s + "\n")
        f.write("# AutoGenerated by KFP." "{{workflow.name}}")
    



def move_mar_model(model_name: str, experiment_name: str, root_path: str, temp_path: str) -> None: # Can be replaced with text component
    import os
    import shutil

    mar_name = "{}.mar".format(model_name)
    model_store_path = os.path.join(root_path, experiment_name, "model-store")
    mar_model = os.path.join(temp_path, mar_name)
    
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
    model_folder = os.path.join(root_path, experiment_name, "outputs", dataset_name)
    model = os.path.join(model_folder, "pytorch_model.bin")
    vocab = os.path.join(model_folder, "vocab.txt")
    config = os.path.join(model_folder, "config.json")

    return model, "{},{}".format(vocab, config) # type: ignore

@kfp.dsl.pipeline(
    name="End to End Hugging Face Topic Classifier - Serving",
    description="The Serving part of the E2E HF Topic Classifier - DEBUG ONLY"
)
def pipeline(experiment_name: str, volume_name: str, dataset_name: str,
    model_version: str = "1.0", randomize_service_suffix: bool = False, use_seed: bool = True, additional_requirements: list = ["anltk","torchserve","transformers",],
    model_serve_name: str = "", model_serve_threads_count: int = 4, model_serve_queue_size: int = 10,
    model_serve_install_dependencies: bool = True, model_serve_is_default: bool = True, model_serve_workers: int = 1, model_serve_workers_max: int = 5,
    model_serve_batch_size: int = 1, model_serve_timeout: int = 120,
): 
    mount_dir = "/store"
    volumetrize = setup_volume(volume_name, mount_dir)

    # Convert this to a utility function
    handler_import_path = "src/handlers/topic_class.py"
    handler_code = open(handler_import_path, encoding='utf-8').read() 

    handler_op = components.func_to_container_op(create_handler)
    handler_task = handler_op(handler_code, experiment_name, mount_dir, additional_requirements)
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
    move_model_task = move_model_op(model_name, experiment_name, mount_dir, out_temp_path)
    move_model_task = volumetrize(move_model_task)
    move_model_task = move_model_task.after(mar_convert_task).after(get_mar_task)
    cacheless_task(move_model_task)

    # Create Config
    create_config_op = components.func_to_container_op(create_mar_config)
    create_config_task = create_config_op(
        experiment_name, mount_dir, dataset_name,
        model_serve_name, model_version, model_serve_threads_count, model_serve_queue_size,
        model_serve_install_dependencies, model_serve_is_default, model_serve_workers, model_serve_workers_max,
        model_serve_batch_size, model_serve_timeout
    )
    create_config_task = volumetrize(create_config_task)


    serve_task = create_serve_task(dataset_name, experiment_name, volume_name, randomize_service_suffix, use_seed)
    serve_task = volumetrize(serve_task)
    serve_task = serve_task.after(move_model_task).after(create_config_task)
    cacheless_task(serve_task)

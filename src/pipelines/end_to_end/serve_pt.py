import json
from typing import NamedTuple, Optional

import kfp
import yaml
from kfp import components
from kfp.components import func_to_container_op
from src.pipelines.common_utils import (add_pvolumes_func, get_volume_by_name,
                                        spec_from_file_format)


def get_run_args(dataset_name, has_test, seq_len, batch_size_dev, learn_rate, epochs, seed, experiment_name) -> NamedTuple(
    "Outputs",
    [
        ("train_args", dict),
        ("extra_args", dict),
    ]
):
    import json
    import os
    args = {"extra": {}}

    args["save_steps"] = 1000
    args["extra"]["overwrite_output_dir"] = ""

    model_name_or_path = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth"
    args["model_name_or_path"] = model_name_or_path
    
    train_file = '/store/datasets/{}/train.json'.format(dataset_name)
    args["train_file"] = train_file
    args["extra"]["do_train"] = ""
    
    
    valid_file = '/store/datasets/{}/valid.json'.format(dataset_name)
    args["extra"]["validation_file"] = valid_file
    args["extra"]["do_eval"] = ""

    if has_test == True or (isinstance(has_test, str) and has_test.lower() in "true1yes"):
        test_file = '/store/datasets/{}/test.json'.format(dataset_name)
        args["extra"]["test_file"] = test_file
        args["extra"]["do_predict"] = ""
        
    
    args["max_seq_length"] = seq_len
    
    args["per_device_train_batch_size"] = batch_size_dev
    
    args["learning_rate"] = learn_rate
    
    args["num_train_epochs"] = epochs
    
    output_dir = '/store/{}/outputs/{}'.format(experiment_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    args["output_dir"] = output_dir
    
    if seed:
        args["extra"]["seed"] = hash(seed) # Hash of int is the same as int, hash of str is int

    # write the args to a file
    with open(os.path.join(output_dir, "{}-{}-best-hps.json".format(experiment_name, dataset_name)), "w") as f:
        json.dump(args, f)


    print("Args:")
    print(args)

    # convert args to string
    # return " ".join("--{} {}".format(k, v) for k, v in args.items())

    return args, args.pop("extra")

def create_tfjob_task(job_name, hyperparams, mount_name):
    tfjob_chief_spec = spec_from_file_format("src/pipelines/yamls/Specs/HFChief.yaml",
        trainParamValues=hyperparams,
        volumeResourceName=mount_name,
    )
    tfjob_worker_spec = spec_from_file_format("src/pipelines/yamls/Specs/HFWorker.yaml",
        trainParamValues=hyperparams,
    )

    tfjob_op = components.load_component_from_file("src/pipelines/yamls/Components/tfjob_launcher.yaml")

    tfjob_task = tfjob_op(
        name=str(job_name) + "-{{workflow.name}}",
        namespace="{{workflow.namespace}}",
        chief_spec=json.dumps(tfjob_chief_spec),
        worker_spec=json.dumps(tfjob_worker_spec),
        tfjob_timeout_minutes=60,
        delete_finished_tfjob=False
    )

    return tfjob_task



def hf_task(args: str, is_print=False):
    file = "src/pipelines/yamls/Components/hf_trainer_internal.yaml"
    if is_print:
        with open(file, "r", encoding="utf-8") as f:
            hfjob_op = yaml.safe_load(f)
            commands: list = hfjob_op["implementation"]["container"]["args"]
            name: str = hfjob_op["name"]
            description = hfjob_op["description"]
            first_arg = "echo " + commands.pop(0)
            commands.insert(0, first_arg)
            description = "Prints the expected arguments and commands for " + name
            name = "Print " + name
            hfjob_op["implementation"]["container"]["args"] = commands
            hfjob_op["name"] = name
            hfjob_op["description"] = description
            hfjob_op = components.load_component_from_text(yaml.dump(hfjob_op))
    else:
        hfjob_op = components.load_component_from_file(file)

    return hfjob_op(params=args)


def convert_run_args_to_str(train_args: dict, extra_args: dict) -> str:
    print("Args")
    print(train_args)
    print("Extra Args")
    print(extra_args)
    e_args = " ".join("--{} {}".format(k, v) if v != "" else "--" + k for k, v in extra_args.items())
    p_args = " ".join("--{} {}".format(k, v) if v != "" else "--" + k for k, v in train_args.items())
    print("Formatted")
    print(p_args)
    print(e_args)

    return "{} {}".format(p_args, e_args)


def convert_run_args(args: dict, extra_args: dict) -> NamedTuple(
    "Outputs",
    [
        ("model", str),
        ("tf", str),
        ("seq", int),
        ("batch", int),
        ("lr", float),
        ("epoch", int),
        ("out", str),
        ("save", int),
        ("extra_args", str)
    ]
    ):
    print("Input Dict")
    print(args)
    return \
        str     (args["model_name_or_path"]),           \
        str     (args["train_file"]),                   \
        int     (args["max_seq_length"]),               \
        int     (args["per_device_train_batch_size"]),  \
        float   (args["learning_rate"]),                \
        int     (args["num_train_epochs"]),             \
        str     (args["output_dir"]),                   \
        int     (args["save_steps"]),                   \
        " ".join("--{} {}".format(k, v) if v is not "" else "--" + k for k, v in extra_args.items())

def create_serve_task(dataset_name, experiment_name, mount_name):
    model_name = "{}_{}".format(dataset_name, experiment_name)
    
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

@kfp.dsl.pipeline(
    name="End to End Hugging Face Topic Classifier",
    description="End to End Topic Classiciation using HuggingFace Framework and CamelBert model"
)
def pipeline(experiment_name: str, volume_name: str, dataset_name: str): 
    mount_vol = get_volume_by_name(volume_name, "volume-bind")
    mount_dict = {"/store": mount_vol}
    volumetrize = add_pvolumes_func(mount_dict)

    serve_task = create_serve_task(dataset_name, experiment_name, volume_name)
    serve_task = volumetrize(serve_task)

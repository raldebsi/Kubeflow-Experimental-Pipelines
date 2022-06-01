from typing import NamedTuple, Optional
import kfp
from kfp.components import func_to_container_op
from kfp import components
import json

import yaml

from src.pipelines.common_utils import add_pvolumes_func, get_or_create_pvc, get_volume_by_name, spec_from_file_format

def get_run_args(dataset_name, has_valid, seq_len, batch_size_dev, learn_rate, epochs, seed, experiment_name) -> NamedTuple(
    "Outputs",
    [
        ("train_args", dict),
        ("extra_args", dict),
    ]
):
    import os
    import json
    args = {"extra": {}}

    args["save_steps"] = 1000
    args["extra"]["overwrite_output_dir"] = ""

    model_name_or_path = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth"
    args["model_name_or_path"] = model_name_or_path
    
    train_file = '/store/datasets/{}/train.json'.format(dataset_name)
    args["train_file"] = train_file
    args["extra"]["do_train"] = ""
    # args.append("--train_file={}".format(train_file))
    
    if has_valid:
        valid_file = '/store/datasets/{}/valid.json'.format(dataset_name)
        args["extra"]["validation_file"] = valid_file
        args["extra"]["do_eval"] = ""
        # args.append("--valid_file={}".format(valid_file))
    
    args["max_seq_length"] = seq_len
    # args.append("--max_seq_length={}".format(seq_len))
    
    args["per_device_train_batch_size"] = batch_size_dev
    # args.append("--per_device_train_batch_size={}".format(batch_size_dev))
    
    args["learning_rate"] = learn_rate
    # args.append("--learning_rate={}".format(learn_rate))
    
    args["num_train_epochs"] = epochs
    # args.append("--num_train_epochs={}".format(epochs))
    
    output_dir = '/store/{}/outputs/{}'.format(experiment_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    args["output_dir"] = output_dir
    # args.append("--output_dir={}".format(output_dir))
    
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



def hf_task(model, train, seq, batch, lr, epoch, out, save, extra_params: dict, is_print=False):
    file = "src/pipelines/yamls/Components/hf_trainer_internal.yaml"
    if is_print:
        with open(file, "r", encoding="utf-8") as f:
            hfjob_op = yaml.safe_load(f)
            commands: list = hfjob_op["implementation"]["container"]["command"]
            name: str = hfjob_op["name"]
            description = hfjob_op["description"]
            commands.pop(0)
            commands.insert(0, "echo")
            description = "Prints the expected arguments and commands for " + name
            name = "Print " + name
            hfjob_op["implementation"]["container"]["command"] = commands
            hfjob_op["name"] = name
            hfjob_op["description"] = description
            hfjob_op = components.load_component_from_text(yaml.dump(hfjob_op))
    else:
        hfjob_op = components.load_component_from_file(file)


        # Stringify
    # for key, val in params.items():
    #     params[key] = str(val)
    # for key, val in extra_params.items():
    #     extra_params[key] = str(val)

    task = hfjob_op(
        model=model,
        train_file=train,
        max_seq_len=seq,
        batch_size=batch,
        learning_rate=lr,
        epochs=epoch,
        save_path=out,
        save_steps=save,
        extra_params=extra_params,
    )

    return task

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


@kfp.dsl.pipeline(
    name="End to End Hugging Face Topic Classifier",
    description="End to End Topic Classiciation using HuggingFace Framework and CamelBert model"
)
def pipeline(experiment_name: str, volume_name: str,
                dataset_name: str, has_valid: bool = False,
                max_sequence_length: int = 512, device_batch_size: int = 8,
                learning_rate: float = 3e-5, epochs: int = 5, seed: Optional[int] = None):

    # mount_v = get_volume_by_name(volume_name)
    volume_name = "%s-%s" % ("{{workflow.name}}", volume_name)
    mount_vol = kfp.dsl.VolumeOp(
        # name="Create Ridhwan Volumes",
        name="Create Volumes %s" % "ExperimentE2E",
        size="1Gi",
        modes=kfp.dsl.VOLUME_MODE_RWO,
        resource_name=volume_name,
        generate_unique_name=False,
    )
    mount_v = mount_vol.volume
    pvolumes = {"/store": mount_v}
    volumetrize = add_pvolumes_func(pvolumes)

    # mount_vol = get_volume_by_name(volume_name)
    # mount_dict = {"/store": mount_vol}
    # mount_vol = get_or_create_pvc("Create Ridhwan Volumes", "4Gi", volume_name)
    # mount_dict = {"/store": mount_vol.volume}
    # volumetrize = add_pvolumes_func(mount_dict)

    
    get_args_comp = func_to_container_op(get_run_args) # Simulate Katib
    get_args_task = get_args_comp(dataset_name, has_valid, max_sequence_length, device_batch_size, learning_rate, epochs, seed, experiment_name)
    get_args_task = volumetrize(get_args_task)

    convert_args_comp = func_to_container_op(convert_run_args)
    convert_args_task = convert_args_comp(
        get_args_task.outputs["train_args"],
        get_args_task.outputs["extra_args"],
    )


    # train_task = create_tfjob_task(experiment_name, convert_args_task.output, volume_name)
    # train_task = volumetrize(train_task)

    print_train_task =  hf_task(
        convert_args_task.outputs["model"],
        convert_args_task.outputs["tf"],
        convert_args_task.outputs["seq"],
        convert_args_task.outputs["batch"],
        convert_args_task.outputs["lr"],
        convert_args_task.outputs["epoch"],
        convert_args_task.outputs["out"],
        convert_args_task.outputs["save"],
        convert_args_task.outputs["extra_args"],
        True
    )

    train_task = hf_task(
        convert_args_task.outputs["model"],
        convert_args_task.outputs["tf"],
        convert_args_task.outputs["seq"],
        convert_args_task.outputs["batch"],
        convert_args_task.outputs["lr"],
        convert_args_task.outputs["epoch"],
        convert_args_task.outputs["out"],
        convert_args_task.outputs["save"],
        convert_args_task.outputs["extra_args"],
    )
    train_task = volumetrize(train_task)

from typing import Optional
import kfp
from kfp.components import func_to_container_op
from kfp import components
import json

from src.pipelines.common_utils import add_pvolumes_func, get_or_create_pvc, get_volume_by_name, spec_from_file_format

def get_run_args(dataset_name, has_valid, seq_len, batch_size_dev, learn_rate, epochs, seed, experiment_name) -> str:
    import os
    import json
    args = {}

    model_name_or_path = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth"
    args["model_name_or_path"] = model_name_or_path
    
    train_file = '/store/datasets/{}/train.json'.format(dataset_name)
    args["train-file"] = train_file
    # args.append("--train_file={}".format(train_file))
    
    if has_valid:
        valid_file = '/store/datasets/{}/valid.json'.format(dataset_name)
        args["valid-file"] = valid_file
        # args.append("--valid_file={}".format(valid_file))
    
    args["max_seq_length-len"] = seq_len
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
    
    try:
        seed = int(seed)
        args["seed"] = seed
        # args.append("--seed={}".format(seed))
    except ValueError:
        pass

    # write the args to a file
    with open(os.path.join(output_dir, "{}-{}-best-hps.json".format(experiment_name, dataset_name)), "w") as f:
        json.dump(args, f)

    # convert args to string
    return " ".join("--{}={}".format(k, v) for k, v in args.items())

def create_tfjob_task(job_name, hyperparams, mount_name):
    tfjob_chief_spec = spec_from_file_format("src/pipelines/yamls/Specs/HFChief.yaml",
        trainParamValues=hyperparams,
        volumeResourceName=mount_name, # Inactive currently since I handle it externally
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


@kfp.dsl.pipeline(
    name="End to End Hugging Face Topic Classifier",
    description="End to End Topic Classiciation using HuggingFace Framework and CamelBert model"
)
def pipeline(experiment_name: str, volume_name: str,
                dataset_name: str, has_valid: bool = False,
                max_sequence_length: int = 512, device_batch_size: int = 8,
                learning_rate: float = 3e-5, epochs: int = 5, seed: Optional[int] = None):
    
    mount_vol = get_volume_by_name(volume_name)
    convert_args_comp = func_to_container_op(get_run_args)
    mount_vol = get_or_create_pvc("Create Ridhwan Volumes", "4Gi", "New Data").volume
    mount_dict = {"/store": mount_vol}
    volumetrize = add_pvolumes_func(mount_dict)

    convert_args_task = convert_args_comp(dataset_name, has_valid, max_sequence_length, device_batch_size, learning_rate, epochs, seed, experiment_name)
    convert_args_task = volumetrize(convert_args_task)

    train_task = create_tfjob_task(experiment_name, convert_args_task.output, volume_name)
    volumetrize(train_task)



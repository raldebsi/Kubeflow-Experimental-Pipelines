from typing import NamedTuple, List
import kfp
import kfp.compiler
import kfp.components

def wget_data(url: str, output_path: str) -> None:
    """
    Download data from a URL.
    """
    import wget
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Downloading {} to {}".format(url, output_path))
    data = wget.download(url, output_path)
    if not data:
        raise ValueError("Failed to download data from {}".format(url))
    
    return

def validate_data(data_paths: List[str], data_root: str = "/data"):
    """
    Validate data.
    """
    import os

    for data_path in data_paths:
        if not data_path.startswith(data_root):
            data_path = os.path.join(data_root, data_path)
        if not os.path.exists(data_path):
            raise ValueError("Data path {} does not exist".format(data_path))
        if not os.path.isfile(data_path):
            raise ValueError("Data path {} is not a file".format(data_path))
        print("Data {} is valid".format(data_path))

    return


@kfp.dsl.pipeline(
    name="Kube Lablelizer Experiment",
    description="Experiment using KubeFlow to run a ML Train & Test pipeline",
)
def kube_exp_ridhwan(train_source: str, valid_source: str, test_data: kfp.components.InputPath("txt")):
    from kfp import dsl

    print("Starting Pipeline Generation")
    print("Source Params:")
    print("Train Source: {}".format(train_source))
    print("Valid Source: {}".format(valid_source))
    print("Test data: {}".format(test_data))

    mount_vol = dsl.VolumeOp(
        name="Mount Volume",
        resource_name="kube-mount-ridhwan",
        size="4Gi",
        modes=dsl.VOLUME_MODE_RWO,
    )

    data_store_path = "/data"
    pvolumes = {data_store_path: mount_vol.volume}

    # print("Volume:", mount_vol)
    # print("Volume Mount:", pvolumes)

    volumetrize = lambda func: func.add_pvolumes(pvolumes)

    print("Generating Components")
    retreive_data_component = kfp.components.create_component_from_func(wget_data, base_image="python:slim-buster", packages_to_install=["wget"])
    get_train_task = volumetrize(retreive_data_component(train_source, "{root}/train.json".format(root=data_store_path)))
    get_valid_task = volumetrize(retreive_data_component(valid_source, "{root}/dev.json".format(root=data_store_path)))

    validate_data_component = kfp.components.create_component_from_func(validate_data, base_image="python:slim-buster")
    validation_task = validate_data_component(["/data/train.json", "/data/dev.json"], "/data")
    validation_task = volumetrize(validation_task)
    validation_task.after(get_train_task, get_valid_task)


    print("Pipeline Created")

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(kube_exp_ridhwan, __file__.rsplit(".", 1)[0] + ".yaml")
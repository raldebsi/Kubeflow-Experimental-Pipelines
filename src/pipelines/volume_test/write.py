from typing import NamedTuple, List
import kfp
import kfp.compiler
import kfp.components

def write_file(contents: str, path: str):
    import os
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, "file.txt")
    with open(full_path, "w", encoding='utf-8') as f:
        f.write(contents)

def read_file(path: str):
    import os
    full_path = os.path.join(path, "file.txt")
    with open(full_path, "r", encoding='utf-8') as f:
        return f.read()

@kfp.dsl.pipeline(
    name="Writing Into PVC Test",
    description="Test writing to a consistent volume",
)
def write_volume_pipeline():
    from kfp import dsl

    mount_vol = dsl.VolumeOp(
        name="Create Ridhwan Volumes", # Volume Unique Name, will be used as identifier to operation.
                                               # If operation exists then it will be reused.
                                               # If operation does not exist then it will be created and the name will be resource_name below.
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWO,
        resource_name="ridhwan-pvc-mount", # PVC Name, will be {pipeline_name}-{id}-resource if name does not exist and generate_unique_name is True.
        # volume_name="ridhwan-personal-volume", # Volume Name, do not use as it will cause the volume to not be found
        # data_source=PipelineParam(name="data_source"), # This way it will create a volume snapshot from param
        generate_unique_name=False,
    )

    data_store_path = "/data"
    pvolumes = {data_store_path: mount_vol.volume}
    volumetrize = lambda func: func.add_pvolumes(pvolumes)

    # Components
    write_volume_component = kfp.components.create_component_from_func(write_file)
    read_volume_component = kfp.components.create_component_from_func(read_file)
    # Tasks
    write_task = volumetrize(write_volume_component("This is a file created by the pipeline into {}".format("/data"), data_store_path))
    read_task = volumetrize(read_volume_component(data_store_path))
    read_task.after(write_task)
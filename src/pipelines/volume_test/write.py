from typing import NamedTuple, List
import kfp
import kfp.compiler
import kfp.components

def write_file(contents: str, path: str):
    import os
    full_path = os.path.join(path, "file.txt")
    with open(full_path, "w", encoding='utf-8') as f:
        f.write(contents)

def read_file(path: str):
    import os
    full_path = os.path.join(path, "file.txt")
    with open(full_path, "r", encoding='utf-8') as f:
        return f.read()

@kfp.dsl.pipeline(
    name="Shared Volume Pipeline",
    description="Test writing to a consistent volume",
)
def write_volume_pipeline(write_path: str):
    from kfp import dsl

    mount_vol = dsl.VolumeOp(
        name="Create Volume", # Operation Name
        size="4Gi",
        modes=dsl.VOLUME_MODE_RWO,
        resource_name="shared-volume-ridhwan-resource", # PVC Name
        volume_name="shared-volume-ridhwan-volume", # Volume Name
        # data_source=PipelineParam(name="data_source"), # This way it will create a volume snapshot from param, which is probably what we need?
    )

    data_store_path = write_path
    pvolumes = {data_store_path: mount_vol.volume}
    volumetrize = lambda func: func.add_pvolumes(pvolumes)

    # Components
    write_volume_component = kfp.components.create_component_from_func(write_file)
    read_volume_component = kfp.components.create_component_from_func(read_file)
    # Tasks
    write_task = volumetrize(write_volume_component("This is a file created by the pipline " + str(write_path), data_store_path))
    read_task = volumetrize(read_volume_component(data_store_path))
    read_task.after(write_task)
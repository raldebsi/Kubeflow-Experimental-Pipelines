from typing import NamedTuple, List
import kfp
import kfp.compiler
import kfp.components

def read_file(path: str):
    import os
    full_path = os.path.join(path, "file.txt")
    with open(full_path, "r", encoding='utf-8') as f:
        return f.read()

@kfp.dsl.pipeline(
    name="Shared Volume Pipeline",
    description="Test reading from a consistent volume",
)
def read_volume_pipeline(data_path: str):
    from kfp import dsl

    mount_vol = dsl.VolumeOp(
        name="Create Volume", # Operation Name
        size="4Gi",
        modes=dsl.VOLUME_MODE_RWO,
        resource_name="shared-volume-ridhwan-resource", # PVC Name
        volume_name="shared-volume-ridhwan-volume", # Volume Name
        # data_source=PipelineParam(name="data_source"), # This way it will create a volume snapshot from param, which is probably what we need?
    )

    data_store_path = data_path
    pvolumes = {data_store_path: mount_vol.volume}
    volumetrize = lambda func: func.add_pvolumes(pvolumes)

    # Components
    read_volume_component = kfp.components.create_component_from_func(read_file)
    # Tasks
    read_task = volumetrize(read_volume_component(data_store_path))
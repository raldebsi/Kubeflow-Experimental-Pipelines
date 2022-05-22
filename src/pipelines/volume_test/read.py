from typing import NamedTuple, List
import kfp
import kfp.compiler
import kfp.components
from kfp.dsl._pipeline_param import sanitize_k8s_name
from kfp.dsl._pipeline_volume import PipelineVolume
from kfp.dsl._container_op import _register_op_handler

def read_file(path: str):
    import os
    full_path = os.path.join(path, "file.txt")
    print("Full path is {}".format(full_path))
    print("Current path is {}".format(os.getcwd()))
    with open(full_path, "r", encoding='utf-8') as f:
        data = f.read()
        print(data)
        return data

@kfp.dsl.pipeline(
    name="Shared Volume Pipeline",
    description="Test reading from a consistent volume",
)
def read_volume_pipeline(data_path: str):
    from kfp import dsl

    # mount_vol = dsl.VolumeOp(
    #     name="Read Ridhwan Volumes",
    #     size="1Gi",
    #     modes=dsl.VOLUME_MODE_RWO,
    #     resource_name="ridhwan-pvc-mount", # PVC Name, will be {pipeline_name}-{id}-resource if name does not exist and generate_unique_name is True.
    #     volume_name="volumesecond",
    #     generate_unique_name=False,
    # )


    # Get volume
    volume_name = "ridhwan-pvc-mount"
    volume_name = sanitize_k8s_name(volume_name)
    unique_volume_name = "{{workflow.name}}-%s" % volume_name
    mount_vol = PipelineVolume(
        name=unique_volume_name, # Parameter name, if found then it will be reused. Therefore it should be unique.
        pvc=volume_name,
        volume=None, # type: ignore 
    )

    data_store_path = data_path
    # pvolumes = {data_store_path: mount_vol.volume}
    pvolumes = {data_store_path: mount_vol}
    volumetrize = lambda func: func.add_pvolumes(pvolumes)

    # Components
    read_volume_component = kfp.components.create_component_from_func(read_file)
    # Tasks
    read_task = volumetrize(read_volume_component(data_store_path))
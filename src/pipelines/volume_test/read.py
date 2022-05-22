import kfp
import kfp.compiler
import kfp.components
from kfp.dsl._pipeline_param import sanitize_k8s_name
from kfp.dsl._pipeline_volume import PipelineVolume
from src.pipelines.common_utils import get_volume_by_name, add_pvolumes_func


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
    # mount_vol = kfp.dsl.VolumeOp(
    #     name="Read Ridhwan Volumes",
    #     size="1Gi",
    #     modes=kfp.dsl.VOLUME_MODE_RWO,
    #     resource_name="ridhwan-pvc-mount", # PVC Name, will be {pipeline_name}-{id}-resource if name does not exist and generate_unique_name is True.
    #     volume_name="volumesecond",
    #     generate_unique_name=False,
    # )

    mount_vol = get_volume_by_name("ridhwan-pvc-mount")
    data_store_path = data_path
    pvolumes = {data_store_path: mount_vol}
    volumetrize = add_pvolumes_func(pvolumes)

    # Components
    read_volume_component = kfp.components.create_component_from_func(read_file)
    # Tasks
    read_task = volumetrize(read_volume_component(data_store_path))

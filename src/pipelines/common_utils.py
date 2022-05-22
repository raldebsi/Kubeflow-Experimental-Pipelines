import kfp
import kfp.compiler
import kfp.components
from kfp.dsl._pipeline_param import sanitize_k8s_name
from kfp.dsl._pipeline_volume import PipelineVolume

def get_volume_by_name(name) -> PipelineVolume:
    # Get volume
    volume_name = sanitize_k8s_name(name)
    unique_volume_name = "{{workflow.name}}-%s" % volume_name
    mount_vol = PipelineVolume(
        name=unique_volume_name, # Parameter name, if found then it will be reused. Therefore it should be unique.
        pvc=volume_name,
        volume=None, # type: ignore 
    )

    return mount_vol

def add_pvolumes_func(pvolumes):
    return lambda func: func.add_pvolumes(pvolumes)
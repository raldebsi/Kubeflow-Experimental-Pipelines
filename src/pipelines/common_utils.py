import kfp
import kfp.compiler
import kfp.components
from kfp.dsl._pipeline_param import sanitize_k8s_name
from kfp.dsl._pipeline_volume import PipelineVolume
from kfp import dsl
import yaml

def spec_from_file_format(yaml_file, **kwargs):
    with open(yaml_file, 'r') as f:
        component_spec = f.read()
        for k, v in kwargs.items():
            component_spec = component_spec.replace('{' + k + '}', str(v))
        return yaml.safe_load(component_spec)

def get_volume_by_name(name, unique_name = "") -> PipelineVolume:
    # Get volume
    name = str(name)
    is_pipeline_name = name.startswith('{{') and name.endswith('}}')
    if not unique_name:
        volume_name = sanitize_k8s_name(name) if not is_pipeline_name else name
        unique_volume_name = "{{workflow.name}}-%s" % volume_name
    else:
        unique_volume_name = unique_name
    mount_vol = PipelineVolume(
        name=unique_volume_name, # Parameter name, if found then it will be reused. Therefore it should be unique.
        pvc=name,
        volume=None, # type: ignore 
    )

    return mount_vol

def get_or_create_pvc(name: str, size_: str, resource: str, randomize: bool = False, mode=dsl.VOLUME_MODE_RWO):
    if not resource and not randomize:
        raise ValueError("Either resource or randomize should be provided")
    return dsl.VolumeOp(
        name=name,  # Operation Unique Name
                    # If operation exists then it will be reused.
                    # If operation does not exist then it will be created and the name will be resource_name below.
        size=size_,
        modes=mode,
        resource_name=resource, # PVC Name, will be {pipeline_name}-{id}-resource if name does not exist and generate_unique_name is True.
        # volume_name="ridhwan-personal-volume", # Volume Name, do not use as it will cause the volume to not be found
        # data_source=PipelineParam(name="data_source"), # This way it will create a volume snapshot from param
        generate_unique_name=randomize,
    )

def add_pvolumes_func(pvolumes):
    return lambda func: func.add_pvolumes(pvolumes)
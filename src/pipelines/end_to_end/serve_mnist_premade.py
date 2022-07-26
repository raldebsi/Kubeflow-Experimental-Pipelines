import kfp
import yaml
from kfp import components
from src.pipelines.common_utils import cacheless_task, spec_from_file_format


def create_serve_task():
   
    infer_service = spec_from_file_format(
        "src/pipelines/yamls/Specs/KServe_MNIST.yaml",
        modelNamespace="{{workflow.namespace}}",
    )

    serve_op = components.load_component_from_file("src/pipelines/yamls/Components/kserve_launcher.yaml")
    serve_task = serve_op(
        action="apply",
        inferenceservice_yaml=yaml.dump(infer_service), 
    ) 
    
    return serve_task

@kfp.dsl.pipeline(
    name="MNIST Example Server",
    description="Serves the public MNIST torchserve example"
)
def pipeline(): 
    serve_task = create_serve_task()
    cacheless_task(serve_task)

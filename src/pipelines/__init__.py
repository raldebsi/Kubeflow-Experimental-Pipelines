import os
import types
from typing import Callable
import kfp.compiler

# Include here pipelines you want to compile. Piplines are imported from src/pipelines/ and are automatically included.
from .kube_exp_ridhwan import kube_exp_ridhwan
from .add_randoms import main_pipeline as add_randoms
from .s3_bucket_test import pipeline as s3_pipeline
from .volume_test.write import write_volume_pipeline
from .volume_test.read import read_volume_pipeline

PIPELINES_LOCATION = "pipelines"
IS_ROOT = os.path.exists("src")
os.makedirs(PIPELINES_LOCATION, exist_ok=True)

def get_pipelines():
    # Retrieve all pipelines from imported modules
    functions = {val for val in globals().values() if isinstance(val, types.FunctionType)}
    pipelines = [val for val in functions if val.__module__.startswith("src.pipelines.")] # Only include pipelines in src/pipelines/
    return pipelines

ALL_PIPELINES = get_pipelines()

def compile_all():
    for pipeline in ALL_PIPELINES:
        compile(pipeline)

def compile(exp: Callable):
    # Only allow when cd is root (aka src is in this directory)
    if not IS_ROOT:
        raise ValueError("Must be run from root directory of project")
    
    module = exp.__module__.split(".", 2)[-1]
    pipeline_name = f"{module}.{exp.__name__}.yaml"
    output_location = os.path.join(PIPELINES_LOCATION, pipeline_name)
    kfp.compiler.Compiler().compile(exp, output_location)
    print("Compiled pipeline to {}".format(output_location))
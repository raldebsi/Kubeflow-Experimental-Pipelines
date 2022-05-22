import os
from typing import Callable
import kfp.compiler
# Include here pipelines you want to compile
from .kube_exp_ridhwan import kube_exp_ridhwan
from .add_randoms import main_pipeline as add_randoms
from .s3_bucket_test import pipeline as s3_pipeline


PIPELINES_LOCATION = "pipelines"
IS_ROOT = os.path.exists("src")
os.makedirs(PIPELINES_LOCATION, exist_ok=True)

def compile_all():
    for pipeline in [kube_exp_ridhwan, add_randoms, s3_pipeline]:
        compile(pipeline)

def compile(exp: Callable):
    # Only allow when cd is root (aka src is in this directory)
    if not IS_ROOT:
        raise ValueError("Must be run from root directory of project")
    
    module = exp.__module__.rsplit(".", 1)[-1]
    pipeline_name = f"{module}.{exp.__name__}.yaml"
    output_location = os.path.join(PIPELINES_LOCATION, pipeline_name)
    kfp.compiler.Compiler().compile(exp, output_location)
    print("Compiled pipeline to {}".format(output_location))
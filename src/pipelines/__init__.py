import os
import types
from typing import Callable
import kfp.compiler
import json
import hashlib

# Include here pipelines you want to compile. Piplines are imported from src/pipelines/ and are automatically included.
from .kube_exp_ridhwan import kube_exp_ridhwan
from .add_randoms import main_pipeline as randpipeline
from .s3_bucket_test import pipeline as s3pipeline
from .volume_test.write import write_volume_pipeline
from .volume_test.read import read_volume_pipeline
from .test_upload import pipeline as testuploadpipline
from .end_to_end.mnist_experiment import pipeline as e2emnistpipeline
from .test_upload import pipeline

PIPELINES_LOCATION = "pipelines"
HASH_LOCATION = os.path.join(PIPELINES_LOCATION, "hash.md5")
IS_ROOT = os.path.exists("src")
os.makedirs(PIPELINES_LOCATION, exist_ok=True)
if os.path.exists(HASH_LOCATION):
    with open(HASH_LOCATION, encoding='utf-8') as f:
        try:
            CODE_HASHES = json.load(f)
        except json.decoder.JSONDecodeError:
            CODE_HASHES = {}
else:
    CODE_HASHES = {}

def hash_code(code):
    return hashlib.md5(code.encode('utf-8')).hexdigest()

def get_pipelines():
    # Retrieve all pipelines from imported modules
    has_changed = False
    functions = {val for val in globals().values() if isinstance(val, types.FunctionType)}
    for val in functions:
        if val.__module__.startswith("src.pipelines."): # Only include pipelines in src/pipelines/
            # Get File Content and hash using md5
            with open(val.__code__.co_filename, encoding='utf-8') as f:
                file_content = f.read()
            file_hash = hash_code(file_content)
            # Check if hash has changed
            lookup_name = "{}.{}".format(val.__module__,val.__name__)
            if file_hash != CODE_HASHES.get(lookup_name, None):
                print("Pipeline {} will be compiled".format(lookup_name))
                has_changed = True
                yield val
                CODE_HASHES[lookup_name] = file_hash 
    
    # Write hash to file
    if has_changed:
        with open(HASH_LOCATION, "w", encoding='utf-8') as f:
            json.dump(CODE_HASHES, f)


# ALL_PIPELINES = set(get_pipelines())

def compile_all():
    for pipeline in get_pipelines():
        compile(pipeline)

def compile(exp: Callable):
    # Only allow when cd is root (aka src is in this directory)
    if not IS_ROOT:
        raise ValueError("Must be run from root directory of project")
    
    module = exp.__module__.split(".", 2)[-1]
    pipeline_name = "{}.{}.yaml".format(module, exp.__name__)
    output_location = os.path.join(PIPELINES_LOCATION, pipeline_name)
    kfp.compiler.Compiler().compile(exp, output_location)
    print("Compiled pipeline to {}".format(output_location))
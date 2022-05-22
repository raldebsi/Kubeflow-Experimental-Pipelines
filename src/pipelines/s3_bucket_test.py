from typing import NamedTuple
import kfp
import kfp.components
import kfp.compiler
import boto3
from kfp.v2.dsl import component, Input, Output, Model

def fake_train(model_path, epochs: int):
    handle = open(model_path, "wb")
    for i in range(int(epochs)):
        handle.write(bytes(i))
    handle.close()

def uploadS3(file, bucket, access_key, secret_key):
    import os
    import boto3
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client = session.client("s3")
    resp = client.upload_file(file, bucket, os.path.basename(file))
    print("Upload Errors:", resp)
    return resp

def connect_s3(access_key, secret_key):
    import boto3
    def list_bucket_dir(bucket):
        objects = [x.key for x in bucket.objects.all()]
        print(objects)
        return objects

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    _s3 = session.resource('s3')
    bucket = _s3.Bucket("kubeflow-poc-saal")
    list_bucket_dir(bucket)
    # upload(session, "s3_pipe.yaml")
    # return bucket

def pipeline(access_key, secret_key, epochs: int = 5):
    base_image = "python:slim-buster"

    component_connectS3 = kfp.components.create_component_from_func(
        connect_s3,
        base_image=base_image,
        packages_to_install=["boto3"],
    )

    component_uploadS3 = kfp.components.create_component_from_func(
        uploadS3,
        base_image=base_image,
        packages_to_install=["boto3"],
    )

    component_train = kfp.components.create_component_from_func(
        fake_train,
        base_image=base_image,
        packages_to_install=["boto3"],

    )

    model_path = "fakeTrain.bin"
    
    ListDirS3Task = component_connectS3(access_key, secret_key)
    FakeTrainTask = component_train(model_path, epochs).after(ListDirS3Task)
    ListDirS3Task2 = component_connectS3(access_key, secret_key).after(FakeTrainTask)
    UploadTaskS3 = component_uploadS3(model_path, "kubeflow-poc-saal", access_key, secret_key).after(FakeTrainTask)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, 's3_pipe.yaml')
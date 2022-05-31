import kfp
from kfp import components, dsl
from kfp.components import func_to_container_op
from kubeflow.katib import (ApiClient, V1beta1AlgorithmSpec,
                            V1beta1ExperimentSpec, V1beta1FeasibleSpace,
                            V1beta1ObjectiveSpec, V1beta1ParameterSpec,
                            V1beta1TrialParameterSpec, V1beta1TrialTemplate)
import yaml

from src.pipelines.common_utils import get_or_create_pvc, spec_from_file_format
import json

def katib_experiment_factory(experiment, namespace, steps):
    max_trial_count = 5
    max_failed_trial_count = 3
    parallel_trial_count = 2

    objective = V1beta1ObjectiveSpec( # Objective is to minimize the loss
        type="minimize",
        goal=0.001,
        objective_metric_name="loss"
    )

    algorithm = V1beta1AlgorithmSpec(
        algorithm_name="random"
    )

    param_learning_rate = V1beta1ParameterSpec(
            name="learning_rate",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(
                min=0.01,
                max=0.05,
        )
    )
    param_learning_rate_spec = V1beta1TrialParameterSpec(
        name="learningRate",
        description="Learning rate for the training model",
        reference="learning_rate",
    )

    param_batch_size = V1beta1ParameterSpec(
        name="batch_size",
        parameter_type="int",
        feasible_space=V1beta1FeasibleSpace(
            min="80",
            max="100",
        )        
    )
    param_batch_size_spec = V1beta1TrialParameterSpec(
        name="batchSize",
        description="Batch size for the training model",
        reference="batch_size",
    )

    params = [
        param_learning_rate,
        param_batch_size        
    ]

    trial_spec = spec_from_file_format("src/pipelines/yamls/Specs/TFJob.yaml", trainStepsParamVal=steps)
    trial_template = V1beta1TrialTemplate(
        primary_container_name="tensorflow",
        trial_parameters=[
            param_learning_rate_spec,
            param_batch_size_spec
        ],
        trial_spec=trial_spec,
    )

    experiment_spec = V1beta1ExperimentSpec(
        max_trial_count=max_trial_count,
        max_failed_trial_count=max_failed_trial_count,
        parallel_trial_count=parallel_trial_count,
        objective=objective,
        algorithm=algorithm,
        parameters=params,
        trial_template=trial_template,
    )

    katib_op = components.load_component_from_file("src/pipelines/yamls/Components/katib_launcher.yaml")
    katib_task = katib_op(
        experiment_name=experiment,
        experiment_namespace=namespace,
        experiment_spec=ApiClient().sanitize_for_serialization(experiment_spec),
        experiment_timeout_minutes=60,
        delete_finished_experiment=False
    )

    return katib_task

def convert_hyperparams(hyperparams) -> str:
    import json
    results = json.loads(hyperparams)
    print("Hyperparams FineTuned: ", results)
    best_params = []
    for param in results["currentOptimalTrial"]["parameterAssignments"]:
        if param["name"] == "learning_rate":
            best_params.append("--tf-learning-rate={}".format(param["value"]))
        elif param["name"] == "batch_size":
            best_params.append("--tf-batch-size={}".format(param["value"]))
    print("Best Params", best_params)
    return " ".join(best_params)

def create_tfjob_task(job_name, job_namespace, steps, hyperparams, mount_name):
    tfjob_chief_spec = spec_from_file_format("src/pipelines/yamls/Specs/TFJobChief.yaml",
        trainingStepsParamVal=steps,
        bestHPsParamVal=hyperparams,
        volumeResourceName=mount_name
    )
    tfjob_worker_spec = spec_from_file_format("src/pipelines/yamls/Specs/TFJobWorker.yaml",
        trainingStepsParamVal=steps,
        bestHPsParamVal=hyperparams,
    )

    tfjob_op = components.load_component_from_file("src/pipelines/yamls/Components/tfjob_launcher.yaml")

    tfjob_task = tfjob_op(
        name=job_name,
        namespace=job_namespace,
        chief_spec=json.dumps(tfjob_chief_spec),
        worker_spec=json.dumps(tfjob_worker_spec),
        tfjob_timeout_minutes=60,
        delete_finished_tfjob=False
    )

    return tfjob_task

def create_serve_task(model_name, model_namespace, mount_name):
    infer_service = spec_from_file_format(
        "src/pipelines/yamls/Specs/KFServe.yaml",
        apiVersion="serving.kubeflow.org/v1beta1",
        modelName=model_name,
        modelNamespace=model_namespace,
        volumeResourceName=mount_name,
    )

    serve_op = components.load_component_from_file("src/pipelines/yamls/Components/kfserve_launcher.yaml")
    serve_task = serve_op(
        action="apply",
        inferenceservice_yaml=yaml.dump(infer_service),
    )
    
    return serve_task


@dsl.pipeline(
    name="MNIST E2E Test",
    description="Testin MNIST end to end pipeline"
)
def pipeline(name="mnist-e2e-test-ridhwan", training_steps="200"):
    name = "{{workflow.name}}-%s" % name
    namespace="{{workflow.namespace}}"
    katib_task = katib_experiment_factory(name, namespace, training_steps)

    volume_pvc = get_or_create_pvc("Create Ridhwan Volumes", "1Gi", "ridhwan-pvc-mount")
    volume_name = volume_pvc.outputs["name"]

    # Convert Params
    convert_params_op = func_to_container_op(convert_hyperparams)
    convert_params_task = convert_params_op(katib_task.output)

    # TFJob
    tfjob_task = create_tfjob_task(name, namespace, training_steps, convert_params_task.output, volume_name)

    # Serve
    serve_task = create_serve_task(name, namespace, volume_name).after(tfjob_task)

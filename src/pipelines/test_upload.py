from typing import NamedTuple
import kfp
import kfp.compiler
import kfp.components

def receive_and_print_data(bin_input: kfp.components.InputPath("DataInput")) -> NamedTuple(
    'Outputs', [
        ("my_out_data", kfp.components.OutputBinaryFile) # ?
    ]
):
    print("Bin_Input: ", bin_input)

    with open(bin_input, encoding='utf-8') as f:
        print("Content:", f.read())
    return "This is my output"

def receive_and_print_any(input):
    print("input", input)
    import os
    if os.path.exists(input):
        print("Is File")
        with open(input, encoding='utf-8') as f:
            print("Data:", f.read())

@kfp.dsl.pipeline(
    "Test File Upload",
    "Testing uploading files from ui",
)
def pipeline(test_data_2: kfp.components.InputPath("DataInput")):
    print(test_data_2)

    comp = kfp.components.create_component_from_func(
        receive_and_print_data,
    )
    comp2 = kfp.components.create_component_from_func(receive_and_print_any)

    task = comp(test_data_2)
    task2 = comp2(task.outputs["my_out_data"])

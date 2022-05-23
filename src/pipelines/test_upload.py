from typing import NamedTuple
import kfp
import kfp.compiler
import kfp.components

def receive_and_print_data(bin_input: kfp.components.InputBinaryFile("DataInput")) -> NamedTuple(
    'Outputs', [
        ("my_out_data", kfp.components.OutputBinaryFile)
    ]
):
    print(bin_input)

    return bin_input

@kfp.dsl.pipeline(
    "Test File Upload",
    "Testing uploading files from ui",
)
def pipeline(test_data_2: kfp.components.InputBinaryFile("DataInput")):
    print(test_data_2)

    comp = kfp.components.create_component_from_func(
        receive_and_print_data,

    )
    task = comp(test_data_2)


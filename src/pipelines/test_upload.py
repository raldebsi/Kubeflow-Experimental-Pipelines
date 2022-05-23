from typing import NamedTuple
import kfp
import kfp.compiler
import kfp.components

def receive_and_print_data(bin_input: kfp.components.InputBinaryFile("myInBin")) -> NamedTuple(
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
def pipeline(test_data_2: kfp.components.InputBinaryFile("myBinData")):
    print(test_data_2)
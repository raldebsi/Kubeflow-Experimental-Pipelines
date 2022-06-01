import kfp
from src.pipelines.common_utils import get_volume_by_name, add_pvolumes_func 

def write_me(to_write: str):
    with open("/store/my_file.txt", "w") as f:
        f.write(to_write)


@kfp.dsl.pipeline(
    "Just Write"
)
def pipeline(volume_name: str, to_write: str):
    mount_vol = get_volume_by_name(volume_name, "my-bind-volume")
    mount_dict = {"/store": mount_vol}
    volumetrize = add_pvolumes_func(mount_dict)

    write_me_op = kfp.components.func_to_container_op(write_me)
    write_me_task = volumetrize(write_me_op(to_write))
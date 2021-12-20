from typing import List, Tuple

import distributed
from dask import delayed
from dask.distributed import Client

from scripts.dask_utils.src_dask_utils.dask_initialization import reconnect_client_to_ssh_cluster


def something_to_do(arg1: int, arg2: int) -> None:
    print(f"foobar {arg1} {arg2}")


def main():

    things_do_do: List[Tuple[int, int]] = [tup for tup in zip(range(0,10), range(11,20))]

    use_dask = True
    if use_dask:
        client: Client = reconnect_client_to_ssh_cluster()
        list_of_computations = []

    for arg1, arg2 in things_do_do:
        if use_dask:
            delayed_func = \
                delayed(something_to_do)(
                    arg1=arg1,
                    arg2=arg2,
                )
            list_of_computations.append(delayed_func)
        else:
            something_to_do(
                arg1=arg1,
                arg2=arg2,
            )
    if use_dask:
        print("start compute")
        print(list_of_computations)
        distributed.wait(client.compute(list_of_computations))
        print("end compute")


if __name__ == '__main__':
    main()


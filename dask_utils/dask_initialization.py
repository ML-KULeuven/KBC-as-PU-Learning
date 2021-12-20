from typing import List

from dask.distributed import Client, SSHCluster
from dask import delayed
import distributed

scheduler_host_pinac22 = 'pinac22'
worker_hosts = [
    # 'pinac22',
    # 'pinac32',
    'pinac33',
    'pinac34',
    'pinac35',
    'pinac36',
    'pinac37',
    'pinac38'
    'pinac39'

]

# scheduler_host = 'pinac40'
# worker_hosts = [
#     'pinac22',
#     'pinac23',
#     'pinac25',
#     'pinac26',
#     'pinac28',
#     'pinac30',
#     'pinac-a',
#     'pinac-b',
# ]

scheduler_host_pinac27 = 'pinac27'
worker_hosts_himecs = [
#     #
#     #     # 'pinac22',
#     #     # 'pinac23',
#     #     # 'pinac24',
#     #     # 'pinac25',
#     #     # 'pinac26',
#     #     # 'pinac27',
#     #     # 'pinac28',
#     #     # 'pinac29',
#     #     # 'pinac30',
#     #     # 'pinac32',
#     #     # 'pinac33',
#     #     # 'pinac34',
#     #     # 'pinac35',
#     #     # 'pinac36',
#     #     # 'pinac37',
#     #     # 'pinac38',
#     #     # 'pinac39',
#     #     # 'pinac40',
    'himec01',
    'himec02',
    # 'himec03',
    # 'himec04',
#     #     # 'pinac-a',
#     #     # 'pinac-b',
#     #     # 'pinac-c',
#     #     # 'pinac-d',
]


# scheduler_host = 'pinac27'
worker_hosts_all = [

    'pinac22',
    'pinac23',
    'pinac24',
    'pinac25',
    'pinac26',
    'pinac27',
    'pinac28',
    'pinac29',
    'pinac30',
    'pinac32',
    'pinac33',
    'pinac34',
    'pinac35',
    'pinac36',
    'pinac37',
    'pinac38',
    'pinac39',
    'pinac40',
    'himec01',
    'himec02',
    'himec03',
    'himec04',
    'pinac-a',
    'pinac-b',
    'pinac-c',
    'pinac-d',
]


def initialize_client_for_ssh_cluster(
        scheduler_host: str,
        worker_hosts: List[str]
) -> Client:
    ssh_hosts = [scheduler_host, *worker_hosts]
    try:
        cluster = SSHCluster(
            hosts=ssh_hosts,
            connect_options={"known_hosts": None},
            worker_options={"nthreads": 1},
            # scheduler_options={"port": 0, "dashboard_address": ":8787"}
        )
        client = Client(cluster)
    except (KeyError, OSError):
        scheduler_address = f'{scheduler_host}:8786'
        client = Client(address=scheduler_address)

    return client


def reconnect_client_to_ssh_cluster(scheduler_host: str) -> Client:
    scheduler_address = f'{scheduler_host}:8786'
    client = Client(address=scheduler_address)

    return client

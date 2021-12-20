from enum import Enum
from logging import Logger
from pprint import pprint
from typing import List, Tuple, Dict, Optional

import distributed
from dask.delayed import Delayed
from distributed import Future, Client

from artificial_bias_experiments.experiment_utils import create_logger, close_logger


class TimeUnitEnum(Enum):
    SECONDS = 1
    MINUTES = 2
    HOURS = 3


class TimeoutUnit:
    def get_timeout_limit_in_seconds(self, duration: float, duration_unit: TimeUnitEnum) -> float:
        if duration_unit == TimeUnitEnum.SECONDS:
            return int(duration)
        elif duration_unit == TimeUnitEnum.MINUTES:
            return int(duration * 60)
        elif duration_unit == TimeUnitEnum.HOURS:
            return int(duration * 3600)


# def compute_delayed_functions_old(
#         list_of_computations: List[Tuple[Delayed, Dict]],
#         client: Client,
#         nb_of_retries_if_erred: int,
#         error_logger_name: str,
#         error_logger_file_name: str
# ) -> None:
#     print("start compute")
#     print(list_of_computations)
#
#     list_of_delayed_function_calls: List[Delayed] = [computation[0] for computation in list_of_computations]
#
#     list_of_futures: List[Future] = client.compute(list_of_delayed_function_calls, retries=nb_of_retries_if_erred)
#     distributed.wait(list_of_futures)
#     print("end compute")
#
#     error_logger: Logger = create_logger(logger_name=error_logger_name, log_file_name=error_logger_file_name)
#     future: Future
#     for future, (delayed, func_args) in zip(list_of_futures, list_of_computations):
#         if future.status == 'error':
#             exception = future.exception()
#             error_logger.error(f"{exception.__class__}: {exception}\n"
#                                f"\tfor arguments {func_args}"
#                                )
#     close_logger(error_logger)


# def compute_delayed_functions_old(
#         list_of_computations: List[Tuple[Delayed, Dict]],
#         client: Client,
#         nb_of_retries_if_erred: int,
#         error_logger_name: str,
#         error_logger_file_name: str,
#         timeout_s: Optional[int]=None
# ) -> None:
#
#     error_logger: Logger = create_logger(logger_name=error_logger_name, log_file_name=error_logger_file_name)
#     try:
#         print("start compute")
#         print(list_of_computations)
#
#         list_of_delayed_function_calls: List[Delayed] = [computation[0] for computation in list_of_computations]
#
#         list_of_futures: List[Future] = client.compute(list_of_delayed_function_calls, retries=nb_of_retries_if_erred)
#         # distributed.as_completed()
#         distributed.wait(list_of_futures)
#         print("end compute")
#         error_logger: Logger = create_logger(logger_name=error_logger_name, log_file_name=error_logger_file_name)
#         future: Future
#         for future, (delayed, func_args) in zip(list_of_futures, list_of_computations):
#             if future.status == 'error':
#                 exception = future.exception()
#                 error_logger.error(f"{exception.__class__}: {exception}\n"
#                                    f"\tfor arguments {func_args}"
#                                    )
#     finally:
#         close_logger(error_logger)


def compute_delayed_functions(
        list_of_computations: List[Tuple[Delayed, Dict]],
        client: Client,
        nb_of_retries_if_erred: int,
        error_logger_name: str,
        error_logger_file_name: str,
        timeout_s: Optional[int]=None
) -> None:
    error_logger: Logger = create_logger(logger_name=error_logger_name, log_file_name=error_logger_file_name)
    try:
        print("start compute")
        print(list_of_computations)

        future_id_to_args_map: Dict[int, Dict] = {}

        list_of_delayed_function_calls: List[Delayed] = []

        computation: Delayed
        arguments: Dict
        for computation, arguments in list_of_computations:
            print("Computation key", computation.key)
            pprint(arguments)
            print("---")
            list_of_delayed_function_calls.append(computation)
            future_id_to_args_map[computation.key] = arguments
            # [computation[0] for computation in list_of_computations]

        list_of_futures: List[Future] = client.compute(list_of_delayed_function_calls, retries=nb_of_retries_if_erred)
        # distributed.as_completed()
        # distributed.wait(list_of_futures)
        completed_future: Future
        for completed_future in distributed.as_completed(list_of_futures):
            func_args = future_id_to_args_map[completed_future.key]
            if completed_future.status == 'error':
                exception = completed_future.exception()

                error_logger.error(f"{exception.__class__}: {exception}\n"
                                   f"\tfor arguments {func_args}"
                                   )
                print(f"Erred computation key {completed_future.key}")
            else:
                print(f"Finished computation key {completed_future.key} corresponding to")
                pprint(func_args)
            print("---")
            del future_id_to_args_map[completed_future.key]
            completed_future.release()

    finally:
        close_logger(error_logger)

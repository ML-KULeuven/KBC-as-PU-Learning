import os


def get_rule_wrapper_filename(
        dir_rule_wrapper: str,
        target_relation: str,
        rule_wrapper_index: int
) -> str:
    filename_rule_wrapper_json: str = os.path.join(
        dir_rule_wrapper,
        f"{target_relation}_rule_wrapper_{str(rule_wrapper_index).zfill(2)}.json"
    )
    return filename_rule_wrapper_json


def get_rule_wrapper_prediction_cache_filename(
        dir_rule_wrapper_prediction_cache: str,
        target_relation:str,
        rule_wrapper_index: int
) -> str:
    filename_rule_wrapper_prediction_cache: str = os.path.join(
        dir_rule_wrapper_prediction_cache,
        f"{target_relation}_rule_wrapper_{str(rule_wrapper_index).zfill(2)}_prediction_cache.csv"
    )
    return filename_rule_wrapper_prediction_cache

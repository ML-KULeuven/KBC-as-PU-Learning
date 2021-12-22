import os
import re
from pathlib import Path
from typing import NamedTuple, List, Union, Optional, Match

import pandas as pd

from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noise_level_to_pu_metrics_controller import \
    RuleWrapperNoiseLevelToPuMetricsMap
from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import RuleWrapper

re_log_growth_rate_pattern = re.compile(r"log_growth_rate([0-1]\.?[0-9]*)")
re_trial_pattern = re.compile(r"trial(\d+)")


class NoisyPropScoresSARPopularityDataFrames(NamedTuple):
    rule_wrappers: pd.DataFrame
    noise_level_to_pu_metrics_map: pd.DataFrame


def get_log_growth_rate_from_string(input_string: str) -> float:
    o_match: Optional[Match[str]] = re.search(re_log_growth_rate_pattern, input_string)
    if o_match is not None:
        log_growth_rate = float(o_match.group(1))
        return log_growth_rate
    else:
        raise Exception(f"No log growth rate found in string {input_string}")


def get_trial_index_from_string(input_string: str) -> int:
    o_match: Optional[Match[str]] = re.search(re_trial_pattern, input_string)
    if o_match is not None:
        random_trial_index = int(o_match.group(1))
        return random_trial_index
    else:
        raise Exception(f"No random trial index found in string {input_string}")


def get_rule_wrapper_row_popularity(
        rule_wrapper: RuleWrapper,
        target_relation: str,
        log_growth_rate: float,
        random_trial_index
):
    partial_row: List[str] = [
        target_relation,
        log_growth_rate,
        random_trial_index
    ]
    partial_row_rule_wrapper: List[str] = rule_wrapper.to_row(include_amie_metrics=False)
    full_row = partial_row + partial_row_rule_wrapper
    return full_row


def get_column_names_df_rule_wrappers_sar_popularity():
    column_names: List[str] = [
                                  'target_relation',
                                  'log_growth_rate',
                                  'random_trial_index'
                              ] + RuleWrapper.get_columns_header_without_amie()
    return column_names


def get_rule_wrappers_as_added_noise_dataframe_tuple_noisy_prop_scores_sar_popularity(
        root_dir_experiment_settings_specific: str,
        target_relation: str,
) -> NoisyPropScoresSARPopularityDataFrames:
    dir_rule_wrappers: str = os.path.join(
        root_dir_experiment_settings_specific,
        'rule_wrappers'
    )
    dir_pu_metrics_of_rule_wrappers: str = os.path.join(
        dir_rule_wrappers,
        "pu_metrics"
    )
    list_of_av_prop_score_data: List[pd.DataFrame] = []
    data_rule_wrappers: List[List[Union[str, float, int]]] = []
    path_dir_rule_wrappers: Path = Path(dir_rule_wrappers)
    path_rule_wrapper: Path
    for path_rule_wrapper in path_dir_rule_wrappers.iterdir():
        if path_rule_wrapper.is_file() and path_rule_wrapper.suffix == '.gz':
            current_rule_wrapper: RuleWrapper = RuleWrapper.read_json(
                str(path_rule_wrapper),
            )
            log_growth_rate = get_log_growth_rate_from_string(str(path_rule_wrapper))

            random_trial_index: int = get_trial_index_from_string(str(path_rule_wrapper))
            data_rule_wrappers.append(
                get_rule_wrapper_row_popularity(
                    rule_wrapper=current_rule_wrapper,
                    target_relation=target_relation,
                    log_growth_rate=log_growth_rate,
                    random_trial_index=random_trial_index
                )
            )

            path_rule_wrapper_stem: str = str(path_rule_wrapper.stem)
            if path_rule_wrapper_stem.endswith(".json"):
                path_rule_wrapper_stem = path_rule_wrapper_stem.split(".json")[0]

            filename_pu_metrics_of_rule_wrapper: str = os.path.join(
                dir_pu_metrics_of_rule_wrappers,
                f"{path_rule_wrapper_stem}.tsv.gz"
            )
            df_pu_metrics: pd.DataFrame = RuleWrapperNoiseLevelToPuMetricsMap.read_csv(
                filename_pu_metrics_of_rule_wrapper
            )
            list_of_av_prop_score_data.append(df_pu_metrics)

    df_rule_wrappers: pd.DataFrame = pd.DataFrame(
        data=data_rule_wrappers,
        columns=get_column_names_df_rule_wrappers_sar_popularity()
    )
    df_rule_wrappers = df_rule_wrappers[
        [
            column_name
            for column_name in df_rule_wrappers.columns
            if column_name not in set(
            [conf.get_name() for conf in ConfidenceEnum.get_propensity_weighted_estimators()]
        )
               and column_name is not ConfidenceEnum.ICW_CONF.get_name()
        ]
    ]
    df_noisy_prop_score_pu_metrics: pd.DataFrame = pd.concat(
        list_of_av_prop_score_data
    )

    return NoisyPropScoresSARPopularityDataFrames(
        rule_wrappers=df_rule_wrappers,
        noise_level_to_pu_metrics_map=df_noisy_prop_score_pu_metrics
    )


def get_rule_wrappers_as_dataframe_noisy_prop_scores_sar_popularity(
        root_dir_experiment_settings_specific: str,
        target_relation: str
) -> pd.DataFrame:
    added_noise_dfs: NoisyPropScoresSARPopularityDataFrames = get_rule_wrappers_as_added_noise_dataframe_tuple_noisy_prop_scores_sar_popularity(
        root_dir_experiment_settings_specific=root_dir_experiment_settings_specific,
        target_relation=target_relation,
    )
    df_rule_wrappers: pd.DataFrame = added_noise_dfs.rule_wrappers
    df_noise_level_to_pu_metrics_map: pd.DataFrame = added_noise_dfs.noise_level_to_pu_metrics_map

    df_rule_wrappers: pd.DataFrame = df_rule_wrappers.merge(
        df_noise_level_to_pu_metrics_map,
        left_on=["Rule", "random_trial_index", "log_growth_rate"],
        right_on=["Rule", "random_trial_index", "log_growth_rate"],
    )

    column_names_logistics: List[str] = [
        'target_relation',
        'log_growth_rate',
        "noise_level",
        'random_trial_index',
        'Rule',
    ]
    df_rule_wrappers = df_rule_wrappers[
        column_names_logistics + [
            col for col in df_rule_wrappers.columns
            if col not in column_names_logistics
        ]
    ]

    return df_rule_wrappers

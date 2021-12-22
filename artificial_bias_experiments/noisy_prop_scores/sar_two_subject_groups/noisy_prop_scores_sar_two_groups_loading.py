from typing import List

import pandas as pd

from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_info import \
    NoisyPropScoresSARExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.image_generation.load_rule_wrappers import \
    load_noisy_prop_scores_sar_two_groups_rule_wrapper_dfs
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.noisy_prop_scores_sar_two_groups_file_naming import \
    NoisyPropScoresSARTwoGroupsFileNamer
from artificial_bias_experiments.noisy_prop_scores.scar.image_generation.load_rule_wrappers import \
    NoisyPropScoresDataFrames


def load_df_noisy_prop_scores_two_groups(
    experiment_info: NoisyPropScoresSARExperimentInfo
) -> pd.DataFrame:
    root_dir_experiment_settings: str = NoisyPropScoresSARTwoGroupsFileNamer.get_dir_experiment_specific(
        experiment_info=experiment_info
    )

    noisy_prop_scores_dfs: NoisyPropScoresDataFrames = load_noisy_prop_scores_sar_two_groups_rule_wrapper_dfs(
        experiment_dir=root_dir_experiment_settings,
        target_relation=experiment_info.target_relation,
        filter_relation=experiment_info.filter_relation,
        filter_group_prop_score=experiment_info.true_prop_scores.in_filter
    )

    df_rule_wrappers = noisy_prop_scores_dfs.rule_wrappers
    df_noisy_prop_scores_to_metrics_map = noisy_prop_scores_dfs.noisy_prop_scores_to_pu_metrics_map

    #######################################################
    # MERGE
    df_rule_wrappers: pd.DataFrame = df_rule_wrappers.merge(
        df_noisy_prop_scores_to_metrics_map,
        left_on=["Rule", "random_trial_index", "prop_score_subj", "prop_score_other"],
        right_on=["Rule", "random_trial_index", "true_prop_scores_in_filter", "true_prop_scores_not_in_filter"],
    ).drop(["prop_score_subj", "prop_score_other"], axis=1)

    column_names_logistics: List[str] = [
        'target_relation',
        'filter_relation',
        'true_prop_scores_in_filter', 'true_prop_scores_not_in_filter',
        'noisy_prop_scores_in_filter', 'noisy_prop_scores_not_in_filter',

        'random_trial_index',
        "Rule"
    ]
    other_columns = [col for col in df_rule_wrappers.columns if col not in column_names_logistics]
    resorted_columns = column_names_logistics + other_columns
    df_rule_wrappers = df_rule_wrappers[resorted_columns]

    return df_rule_wrappers

from pathlib import Path
from typing import List

import pandas as pd

from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    KnownPropScoresSARExperimentInfo
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.column_names import \
    CNameEnum
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.trial_conditional_confidence_stats import \
    TrialConditionalTrueConfidencesManager
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


def get_dataframe_with_info_about_known_prop_scores_sar_two_groups__for(
        dataset_name: str,
        target_relation: str,
        filter_relation: str,
        propensity_score_subjects_of_filter_relation: float,
        propensity_score_other_entities_list: List[float],
        is_pca_version: bool
) -> pd.DataFrame:
    dfs_to_concatenate: List[pd.DataFrame] = []

    prop_score_other_group: float
    for prop_score_other_group in propensity_score_other_entities_list:
        experiment_info = KnownPropScoresSARExperimentInfo(
            dataset_name=dataset_name,
            target_relation=target_relation,
            filter_relation=filter_relation,
            true_prop_scores=PropScoresTwoSARGroups(
                in_filter=propensity_score_subjects_of_filter_relation,
                other=prop_score_other_group
            ),
            is_pca_version=is_pca_version
        )

        experiment_dir_specific: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_experiment_specific(
            experiment_info=experiment_info
        )
        path_experiment_dir_specific: Path = Path(experiment_dir_specific)
        # print(experiment_dir_specific)
        for descendant_path in path_experiment_dir_specific.glob('**/*'):
            if descendant_path.is_file() and descendant_path.name.startswith("conditional_conf_rule_stats_"):
                df_single_random_trial_and_other_prop_score: pd.DataFrame \
                    = TrialConditionalTrueConfidencesManager.read_csv(
                        filename=str(descendant_path)
                    )
                dfs_to_concatenate.append(df_single_random_trial_and_other_prop_score)
    df_with_group_info: pd.DataFrame = pd.concat(dfs_to_concatenate)

    df_with_group_info = df_with_group_info.drop(
        columns=[
            'random_trial_index',
            'true_prop_scores_in_filter',
            'true_prop_scores_not_in_filter',
        ]
    )

    df_with_group_info = df_with_group_info.drop_duplicates()

    # Add extra columns to the DF ####################################
    # Relative confidence columns ########################
    add_relative_confidence_columns_to_df_with_group_info(df_with_group_info)

    # Relative # predictions columns #####################
    add_relative_n_predictions_columns_to_df_with_group_info(df_with_group_info)
    return df_with_group_info


def add_relative_confidence_columns_to_df_with_group_info(df_with_group_info: pd.DataFrame) -> pd.DataFrame:
    df_with_group_info[CNameEnum.cname_rel_true_conf_in_filter.value] = (
            (
                    df_with_group_info[CNameEnum.cname_true_conf_in_filter.value]
                    - df_with_group_info[CNameEnum.cname_true_conf.value]
            )
            / df_with_group_info[CNameEnum.cname_true_conf.value]
    )
    df_with_group_info[CNameEnum.cname_rel_true_conf_not_in_filter.value] = (
            (
                    df_with_group_info[CNameEnum.cname_true_conf_not_in_filter.value]
                    - df_with_group_info[CNameEnum.cname_true_conf.value]
            )
            / df_with_group_info[CNameEnum.cname_true_conf.value]
    )
    return df_with_group_info


def add_relative_n_predictions_columns_to_df_with_group_info(df_with_group_info: pd.DataFrame) -> pd.DataFrame:
    df_with_group_info[CNameEnum.cname_rel_n_preds_in_filter.value] = (
            df_with_group_info[CNameEnum.cname_n_preds_in_filter.value]
            / df_with_group_info[CNameEnum.cname_n_preds.value] * 100
    )
    df_with_group_info[CNameEnum.cname_rel_n_preds_not_in_filter.value] = (
            df_with_group_info[CNameEnum.cname_n_preds_not_in_filter.value]
            / df_with_group_info[CNameEnum.cname_n_preds.value] * 100
    )
    return df_with_group_info

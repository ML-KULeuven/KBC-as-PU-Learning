from typing import NamedTuple

import pandas as pd
from tabulate import tabulate

from kbc_pul.observed_data_generation.abstract_triple_selection import ObservedTargetRelationInfo
from kbc_pul.observed_data_generation.sar_two_subject_groups.subject_based_propensity_score_controller import \
    SetInclusionPropensityScoreController


class SARTwoGroupsObservedTargetRelStats(NamedTuple):
    total_n_ground_truth_literals: int
    total_n_observed_literals: int
    n_observed_lits_in_filter: int
    n_observed_lits_not_in_filter: int
    observed_label_frequency: float
    mask_observed_literals_in_filter_set: pd.Series


def calculated_sar_two_groups_observed_target_rel_stats(
        df_ground_truth_target_relation: pd.DataFrame,
        observed_target_rel: ObservedTargetRelationInfo,
        prop_score_controller: SetInclusionPropensityScoreController,
) -> SARTwoGroupsObservedTargetRelStats:
    """
    Look at how the observed target relation is divided in two groups by the filter relation


    :param df_ground_truth_target_relation:
    :param observed_target_rel:
    :param prop_score_controller:
    :return:
    """
    n_true_target_rel_lits: int = len(df_ground_truth_target_relation)

    mask_observed_literals_in_filter_set: pd.Series = prop_score_controller.get_mask_of_literals_in_set(
        observed_target_rel.df)

    n_observed_target_rel_lits: int = len(observed_target_rel.df)
    n_observed_target_rels_in_filter_set = int(mask_observed_literals_in_filter_set.sum())
    n_observed_target_rels_not_in_filter_set = n_observed_target_rel_lits - n_observed_target_rels_in_filter_set
    observed_label_frequency: float = prop_score_controller.get_label_frequency_given_mask_observed_literals_in_set(
        mask_observed_literals_in_filter_set
    )

    return SARTwoGroupsObservedTargetRelStats(
        total_n_ground_truth_literals=n_true_target_rel_lits,
        total_n_observed_literals=n_observed_target_rel_lits,
        n_observed_lits_in_filter=n_observed_target_rels_in_filter_set,
        n_observed_lits_not_in_filter=n_observed_target_rels_not_in_filter_set,
        observed_label_frequency=observed_label_frequency,
        mask_observed_literals_in_filter_set=mask_observed_literals_in_filter_set
    )


def print_sar_two_groups_observed_target_rel_stats(
        filter_relation: str,
        observed_target_rel_stats: SARTwoGroupsObservedTargetRelStats) -> None:
    observed_to_ground_truth_fraction: float = (
            observed_target_rel_stats.total_n_observed_literals
            /
            observed_target_rel_stats.total_n_ground_truth_literals
    )

    observed_in_filter_to_total_observed_fraction: float = (
            observed_target_rel_stats.n_observed_lits_in_filter
            /
            observed_target_rel_stats.total_n_observed_literals
    )

    not_observed_in_filter_to_total_observed_fraction: float = (
            observed_target_rel_stats.n_observed_lits_not_in_filter
            /
            observed_target_rel_stats.total_n_observed_literals
    )

    table = [
        ["# true target lits:", observed_target_rel_stats.total_n_ground_truth_literals],
        ["# observed target lits:", observed_target_rel_stats.total_n_observed_literals,
         f"({observed_to_ground_truth_fraction * 100 : 0.2f} %)"],
        [f"# observed target lits given {filter_relation}",
         observed_target_rel_stats.n_observed_lits_in_filter,
         f"({observed_in_filter_to_total_observed_fraction * 100 :0.2f} % observed)"
         ],
        [f"# observed target lits given NOT {filter_relation}",
         observed_target_rel_stats.n_observed_lits_not_in_filter,
         f"({not_observed_in_filter_to_total_observed_fraction * 100 :0.2f} % observed)"
         ]
    ]
    print(tabulate(table))

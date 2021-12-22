from typing import NamedTuple, List, Union, Optional, Tuple

import pandas as pd
from tabulate import tabulate

from artificial_bias_experiments.evaluation.ground_truth_utils import TrueEntitySetsTuple
from kbc_pul.observed_data_generation.sar_two_subject_groups.subject_based_propensity_score_controller import \
    SetInclusionPropensityScoreController
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_true_confidence_on_observed_data_from_cached_predictions import \
    get_true_confidence_on_observed_data_using_cached_predictions, \
    get_true_pca_confidence_on_observed_data_using_cached_predictions


class ConditionedTrueConfidenceTuple(NamedTuple):
    n_predictions_in_filter: int
    true_conf_on_predictions_in_filter: float

    n_predictions_not_in_filter: int
    true_conf_on_predictions_not_in_filter: float

    # true_pos_pair_conf_s_to_o_in_filter: float
    # true_pos_pair_conf_s_to_o_not_in_filter: float
    #
    # true_pos_pair_conf_o_to_s_in_filter: float
    # true_pos_pair_conf_o_to_s_not_in_filter: float

    def is_interesting(self) -> bool:
        total_n_predictions = self.n_predictions_in_filter + self.n_predictions_not_in_filter
        return (
                self.n_predictions_in_filter > 10
                and
                self.n_predictions_not_in_filter > 10
                and (self.n_predictions_in_filter / total_n_predictions) > 0.1
                and (self.n_predictions_not_in_filter / total_n_predictions) > 0.1
                and (self.true_conf_on_predictions_in_filter - self.true_conf_on_predictions_not_in_filter) > 0.1
        )

    @staticmethod
    def get_n_predictions_column_names() -> List[str]:
        return [
                   "n_predictions_in_filter",
                   "n_predictions_not_in_filter",
               ]


    @staticmethod
    def get_confidence_column_names(o_conf_prefix: Optional[str] = None) -> List[str]:
        if o_conf_prefix is None:
            o_conf_prefix = ""

        return [
            o_conf_prefix + "conf_on_predictions_in_filter",
            o_conf_prefix + "conf_on_predictions_not_in_filter"
        ]

    @staticmethod
    def get_column_names(o_conf_prefix: Optional[str] = None) -> List[str]:

        if o_conf_prefix is None:
            o_conf_prefix = ""

        return ConditionedTrueConfidenceTuple.get_confidence_column_names(
            o_conf_prefix) + ConditionedTrueConfidenceTuple.get_n_predictions_column_names()

    def to_row(self) -> List[Union[float, int]]:
        return [
            self.true_conf_on_predictions_in_filter,
            self.true_conf_on_predictions_not_in_filter,

            # self.true_pos_pair_conf_s_to_o_in_filter,
            # self.true_pos_pair_conf_s_to_o_not_in_filter,
            #
            # self.true_pos_pair_conf_o_to_s_in_filter,
            # self.true_pos_pair_conf_o_to_s_not_in_filter,

            self.n_predictions_in_filter,
            self.n_predictions_not_in_filter,
        ]


def get_true_confidences_conditioned_on_filter_set(
        o_df_cached_predictions: pd.DataFrame,
        set_incl_prop_score_controller: SetInclusionPropensityScoreController,
        df_ground_truth_target_rel: pd.DataFrame,
) -> Optional[ConditionedTrueConfidenceTuple]:
    if o_df_cached_predictions is None or len(o_df_cached_predictions) == 0:
        raise Exception("DataFrame is None or empty")

    mask_of_cached_predictions_in_filter_set: pd.Series = set_incl_prop_score_controller.get_mask_of_literals_in_set(
        o_df_cached_predictions[["Subject", "Object"]]
    )
    df_cached_predictions_in_filters_set: pd.DataFrame = o_df_cached_predictions[
        mask_of_cached_predictions_in_filter_set
    ]
    df_cached_predictions_not_in_filter_set: pd.DataFrame = o_df_cached_predictions[
        ~mask_of_cached_predictions_in_filter_set
    ]

    n_predictions_in_filter = len(df_cached_predictions_in_filters_set)
    n_predictions_not_in_filter = len(df_cached_predictions_not_in_filter_set)

    if n_predictions_in_filter == 0 or n_predictions_not_in_filter == 0:
        return None
    else:
        true_conf_predictions_in_filter_set: float = get_true_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions_in_filters_set,
            df_ground_truth_target_relation=df_ground_truth_target_rel
        )
        true_conf_predictions_not_in_filter_set: float = get_true_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions_not_in_filter_set,
            df_ground_truth_target_relation=df_ground_truth_target_rel,
        )
        return ConditionedTrueConfidenceTuple(
            true_conf_on_predictions_in_filter=true_conf_predictions_in_filter_set,
            true_conf_on_predictions_not_in_filter=true_conf_predictions_not_in_filter_set,

            n_predictions_in_filter=n_predictions_in_filter,
            n_predictions_not_in_filter=n_predictions_not_in_filter,
        )


def get_true_positive_pair_confidence_on_filter_set(
        o_df_cached_predictions: pd.DataFrame,
        set_incl_prop_score_controller: SetInclusionPropensityScoreController,
        true_entity_sets: TrueEntitySetsTuple,
) -> Optional[
        Tuple[ConditionedTrueConfidenceTuple, ConditionedTrueConfidenceTuple]
]:
    if o_df_cached_predictions is None or len(o_df_cached_predictions) == 0:
        raise Exception("DataFrame is None or empty")

    mask_of_cached_predictions_in_filter_set: pd.Series = set_incl_prop_score_controller.get_mask_of_literals_in_set(
        o_df_cached_predictions[["Subject", "Object"]]
    )
    df_cached_predictions_in_filters_set: pd.DataFrame = o_df_cached_predictions[
        mask_of_cached_predictions_in_filter_set
    ]
    df_cached_predictions_not_in_filter_set: pd.DataFrame = o_df_cached_predictions[
        ~mask_of_cached_predictions_in_filter_set
    ]

    n_predictions_in_filter = len(df_cached_predictions_in_filters_set)
    n_predictions_not_in_filter = len(df_cached_predictions_not_in_filter_set)

    if n_predictions_in_filter == 0 or n_predictions_not_in_filter == 0:
        return None
    else:
        true_pca_conf_s_to_o_in_filter_set: float = get_true_pca_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions_in_filters_set,
            true_entity_str_tuple_set=true_entity_sets.entity_pairs,
            true_pca_non_target_entity_set=true_entity_sets.pca_subjects,
            predict_object_entity=True,
        )
        true_pca_conf_s_to_o_not_in_filter_set: float = get_true_pca_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions_not_in_filter_set,
            true_entity_str_tuple_set=true_entity_sets.entity_pairs,
            true_pca_non_target_entity_set=true_entity_sets.pca_subjects,
            predict_object_entity=True,
        )

        true_pca_conf_o_to_s_in_filter_set: float = get_true_pca_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions_in_filters_set,
            true_entity_str_tuple_set=true_entity_sets.entity_pairs,
            true_pca_non_target_entity_set=true_entity_sets.pca_objects,
            predict_object_entity=False
        )

        true_pca_conf_o_to_s_not_in_filter_set: float = get_true_pca_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions_not_in_filter_set,
            true_entity_str_tuple_set=true_entity_sets.entity_pairs,
            true_pca_non_target_entity_set=true_entity_sets.pca_objects,
            predict_object_entity=False
        )

        s_to_o_tuple = ConditionedTrueConfidenceTuple(

            true_conf_on_predictions_in_filter=true_pca_conf_s_to_o_in_filter_set,
            true_conf_on_predictions_not_in_filter=true_pca_conf_s_to_o_not_in_filter_set,

            n_predictions_in_filter=n_predictions_in_filter,
            n_predictions_not_in_filter=n_predictions_not_in_filter,

        )

        o_to_s_tuple = ConditionedTrueConfidenceTuple(

            true_conf_on_predictions_in_filter=true_pca_conf_o_to_s_in_filter_set,
            true_conf_on_predictions_not_in_filter=true_pca_conf_o_to_s_not_in_filter_set,

            n_predictions_in_filter=n_predictions_in_filter,
            n_predictions_not_in_filter=n_predictions_not_in_filter,
        )
        return s_to_o_tuple, o_to_s_tuple


def print_conditioned_true_confidences(
        filter_relation: str,
        conditioned_true_confidence_tuple: ConditionedTrueConfidenceTuple,
        true_confidence_on_all_predictions: float,
        # observed_target_rel_stats: SARTwoGroupsObservedTargetRelStats
):
    total_n_predicted_lits = (
            conditioned_true_confidence_tuple.n_predictions_in_filter
            +
            conditioned_true_confidence_tuple.n_predictions_not_in_filter
    )

    table = [
        [f"True conf conditioned on {filter_relation}:",
         f"{conditioned_true_confidence_tuple.true_conf_on_predictions_in_filter: 0.3f}",
         f"on {conditioned_true_confidence_tuple.n_predictions_in_filter} predictions ",
         f"{conditioned_true_confidence_tuple.n_predictions_in_filter / total_n_predicted_lits * 100: 0.2f} %"

         # f" {observed_target_rel_stats.n_observed_lits_in_filter} lits"
         ],

        [f"True conf condition on NOT {filter_relation}: ",
         f"{conditioned_true_confidence_tuple.true_conf_on_predictions_not_in_filter: 0.3f}",
         f"on {conditioned_true_confidence_tuple.n_predictions_not_in_filter} predictions",
         f"{conditioned_true_confidence_tuple.n_predictions_not_in_filter / total_n_predicted_lits * 100: 0.2f} %"
         ],

        [f"True conf on all predictions:",
         f" {true_confidence_on_all_predictions: 0.3}",
         f"on {total_n_predicted_lits} predictions",
         "100%"
         ]
    ]
    print(tabulate(table))
    print()

from typing import Set, Tuple, Optional

import pandas as pd

from artificial_bias_experiments.evaluation.ground_truth_utils import TrueEntitySetsTuple
from kbc_pul.data_structures.rule_wrapper import RuleWrapper


def get_true_confidence_on_observed_data_using_cached_predictions(
        df_cached_predictions: pd.DataFrame,
        df_ground_truth_target_relation: pd.DataFrame,
) -> float:
    n_predictions: int = len(df_cached_predictions)

    n_true_predictions: int = df_cached_predictions.merge(
        right=df_ground_truth_target_relation,
        left_on=["Subject", "Object"],
        right_on=["Subject", "Object"],

    ).drop_duplicates().shape[0]
    n_true_predictions: int = int(n_true_predictions)
    return n_true_predictions / n_predictions


def get_true_pca_confidence_on_observed_data_using_cached_predictions(
        df_cached_predictions: pd.DataFrame,
        true_entity_str_tuple_set: Set[Tuple[str, str]],
        true_pca_non_target_entity_set: Set[str],
        predict_object_entity: bool,
        verbose: bool = False
) -> Optional[float]:
    """
    Given a rule r: h(X,Y) <= B,

    """
    if predict_object_entity:
        non_target_index = 0
    else:
        non_target_index = 1

    mask_true_positives: pd.Series = df_cached_predictions.apply(
        lambda row: (row["Subject"], row["Object"]) in true_entity_str_tuple_set,
        axis=1,
        result_type='reduce'
    )
    n_known_positive_predictions: int = int(mask_true_positives.sum())

    n_pca_negative_predictions: int = df_cached_predictions[~mask_true_positives].apply(
        lambda row: (row["Subject"], row["Object"])[non_target_index] in true_pca_non_target_entity_set,
        axis=1,
        result_type='reduce'
    ).sum()
    n_pca_unknown_predictions: int = len(df_cached_predictions) - n_pca_negative_predictions - n_known_positive_predictions
    if verbose:
        print(f"target : {'object' if predict_object_entity else 'subject'}")
        print(f"# KPs : {n_known_positive_predictions}")
        print(f"# PCA negatives : {n_pca_negative_predictions}")
        print(f"# PCA unknowns: {n_pca_unknown_predictions}")

    pca_body_support: int = n_known_positive_predictions + n_pca_negative_predictions

    if pca_body_support == 0:
        print(f"PCA body support is 0 (for {len(df_cached_predictions)} predictions)")
        print("--> cannot compute PCA confidence on observed data")
        return None
    else:
        pca_confidence: float = n_known_positive_predictions / pca_body_support
        return pca_confidence


EntityStr = str


def calculate_true_confidence_metrics_from_df_cached_predictions(
        rule_wrapper: RuleWrapper,
        df_cached_predictions: pd.DataFrame,
        df_ground_truth_target_relation: pd.DataFrame,
        true_entity_sets: TrueEntitySetsTuple,
        verbose: bool = False
) -> None:
    if len(df_cached_predictions) != 0:

        # TRUE CONF
        true_conf: float = get_true_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions,
            df_ground_truth_target_relation=df_ground_truth_target_relation,

        )
        rule_wrapper.o_true_confidence = true_conf

        # TRUE pair-positive confidence ('conf*') S->O
        true_pca_conf_subject_to_object: float = get_true_pca_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions,
            true_entity_str_tuple_set=true_entity_sets.entity_pairs,
            true_pca_non_target_entity_set=true_entity_sets.pca_subjects,
            predict_object_entity=True,
        )
        rule_wrapper.o_true_pca_confidence_subject_to_object = true_pca_conf_subject_to_object

        # TRUE pair-positive confidence ('conf*') O->S
        true_pca_conf_object_to_subject: float = get_true_pca_confidence_on_observed_data_using_cached_predictions(
            df_cached_predictions=df_cached_predictions,
            true_entity_str_tuple_set=true_entity_sets.entity_pairs,
            true_pca_non_target_entity_set=true_entity_sets.pca_objects,
            predict_object_entity=False
        )
        rule_wrapper.o_true_pca_confidence_object_to_subject = true_pca_conf_object_to_subject
    else:
        pass  # no predictions

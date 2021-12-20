from typing import Optional

import pandas as pd

from kbc_pul.data_structures.rule_wrapper import RuleWrapper

# cached_predictions_column_names = [
#     "Subject",
#     "Object",
#     "is_supported",
#     "exists_lits_same_subject",
#     "exists_lits_same_object"
# ]


def set_rule_wrapper_metrics_from_cache(
        rule_wrapper: RuleWrapper,
        df_cached_predictions: pd.DataFrame
) -> None:
    n_predictions = int(len(df_cached_predictions))

    rule_wrapper.o_n_predictions = n_predictions
    if n_predictions > 0:
        rule_wrapper.o_n_supported_predictions = int(df_cached_predictions['is_supported'].sum())

        rule_wrapper.o_std_confidence = calculate_cwa_confidence_from_df_cache(df_cached_predictions)

        rule_wrapper.o_pca_confidence_subject_to_object = calculate_pca_confidence_s_to_o_from_df_cache(
            df_cached_predictions)
        rule_wrapper.o_pca_confidence_object_to_subject = calculate_pca_confidence_o_to_s_from_df_cache(
            df_cached_predictions)


def calculate_cwa_confidence_from_df_cache(df_cached_predictions: pd.DataFrame) -> Optional[float]:
    if len(df_cached_predictions) == 0:
        return None
    cached_n_supported_predictions = df_cached_predictions['is_supported'].sum()
    cached_cwa_conf = cached_n_supported_predictions / len(df_cached_predictions)
    return float(cached_cwa_conf)


def calculate_pca_confidence_s_to_o_from_df_cache(df_cached_predictions: pd.DataFrame) -> Optional[float]:
    if len(df_cached_predictions) == 0:
        return None

    cached_n_pca_negs_s_to_o = (
            (~df_cached_predictions['is_supported']) &
            df_cached_predictions['exists_lits_same_subject']
    ).sum()

    cached_n_supported_predictions = df_cached_predictions['is_supported'].sum()
    cached_pca_s_to_o = cached_n_supported_predictions / (
            cached_n_supported_predictions + cached_n_pca_negs_s_to_o
    )
    return float(cached_pca_s_to_o)


def calculate_pca_confidence_o_to_s_from_df_cache(df_cached_predictions: pd.DataFrame) -> Optional[float]:
    if len(df_cached_predictions) == 0:
        return None

    cached_n_pca_negs_o_to_s = (
            ~df_cached_predictions['is_supported']
            & df_cached_predictions['exists_lits_same_object']
    ).sum()
    cached_n_supported_predictions = df_cached_predictions['is_supported'].sum()

    cached_pca_body = cached_n_supported_predictions + cached_n_pca_negs_o_to_s
    if cached_pca_body == 0:
        return None
    else:
        cached_pca_o_to_s = cached_n_supported_predictions / cached_pca_body
        return float(cached_pca_o_to_s)

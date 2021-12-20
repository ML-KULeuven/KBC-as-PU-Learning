from typing import Sequence, Optional

import pandas as pd
from pylo.language.lp import Atom as PyloAtom
from pylo.language.lp import Variable as PyloVariable, Context as PyloContext

from kbc_pul.rule_metrics.prolog_kb_rule_metrics.rule_pu_confidence_unbiased import get_inverse_propensity_score_for_literal


def get_inverse_propensity_score_for_literal_from_row(
        row: pd.Series,
        rule_head: PyloAtom,
        pylo_context: PyloContext,
        propensity_score_controller,
        subject_var: PyloVariable,
        object_var: PyloVariable
) -> float:
    subject_str: str = row["Subject"]
    object_str: str = row["Object"]

    predicted_literal: PyloAtom = rule_head.substitute(
        {
            subject_var: pylo_context.constant(subject_str),
            object_var: pylo_context.constant(object_str)
        }
    )
    # weight the literal with its inverse propensity score
    inverse_propensity_score_for_literal: float = get_inverse_propensity_score_for_literal(
        predicted_literal=predicted_literal,
        propensity_score_controller=propensity_score_controller,
        propensity_score_literal_counter=None
    )
    return inverse_propensity_score_for_literal


def get_inverse_propensity_score_per_prediction(
        df_predictions_to_weight: pd.DataFrame,
        rule_head: PyloAtom,
        pylo_context: PyloContext,
        propensity_score_controller
) -> pd.Series:
    head_variables: Sequence[PyloVariable] = rule_head.get_variables()

    subject_var: PyloVariable = head_variables[0]
    object_var: PyloVariable = head_variables[1]

    row: pd.Series
    series_inverse_prop_scores: pd.Series = df_predictions_to_weight.apply(
        lambda row: get_inverse_propensity_score_for_literal_from_row(
            row=row,
            rule_head=rule_head,
            pylo_context=pylo_context,
            propensity_score_controller=propensity_score_controller,
            subject_var=subject_var,
            object_var=object_var
        ),
        axis=1,
        result_type='reduce'
    )
    return series_inverse_prop_scores


def get_inverse_propensity_weighted_count_of_predictions(
        df_predictions_to_weight: pd.DataFrame,
        rule_head: PyloAtom,
        pylo_context: PyloContext,
        propensity_score_controller
):
    if len(df_predictions_to_weight) == 0:
        return 0

    # propensity_weighted_count: float = 0.0
    row: pd.Series
    series_inverse_prop_scores: pd.Series = get_inverse_propensity_score_per_prediction(
        df_predictions_to_weight=df_predictions_to_weight,
        rule_head=rule_head,
        pylo_context=pylo_context,
        propensity_score_controller=propensity_score_controller
    )
    propensity_weighted_count: float = float(series_inverse_prop_scores.sum())

    return propensity_weighted_count


def calculate_pu_propensity_confidence_unbiased_from_df_cache(
        df_cached_predictions: pd.DataFrame,
        rule_head: PyloAtom,
        pylo_context: PyloContext,
        propensity_score_controller,
        verbose: bool = False,
        o_propensity_score_per_prediction: Optional[pd.Series] = None
) -> Optional[float]:
    n_predictions: int = len(df_cached_predictions)
    if n_predictions > 0:
        mask_supported_predictions = df_cached_predictions['is_supported']
        if o_propensity_score_per_prediction is None:
            df_known_positives: pd.DataFrame = df_cached_predictions[
                mask_supported_predictions
                ]

            weighted_known_positives: float = get_inverse_propensity_weighted_count_of_predictions(
                df_predictions_to_weight=df_known_positives,
                rule_head=rule_head,
                pylo_context=pylo_context,
                propensity_score_controller=propensity_score_controller
            )
        else:
            weighted_known_positives: float = float(o_propensity_score_per_prediction[mask_supported_predictions].sum())

        if verbose and weighted_known_positives > n_predictions:
            print(f"WARNING: e-conf est > 1: {weighted_known_positives} / {n_predictions}")

        pu_confidence: float = weighted_known_positives / n_predictions
        return pu_confidence
    else:
        return None

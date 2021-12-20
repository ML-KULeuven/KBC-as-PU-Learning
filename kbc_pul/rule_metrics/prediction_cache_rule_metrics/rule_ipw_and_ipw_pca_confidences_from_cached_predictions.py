import pandas as pd
from pylo.language.lp import (Atom as PyloAtom, Context as PyloContext)

from kbc_pul.data_structures.rule_wrapper import RuleWrapper
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_ipw_pca_confidence_from_cached_predictions import \
    calculate_inverse_propensity_weighted_pca_confidence_from_df_cache
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_ipw_confidence_from_cached_predictions import \
    calculate_inverse_propensity_weighted_confidence_from_df_cache, get_inverse_propensity_weighted_count_of_predictions, \
    get_inverse_propensity_score_per_prediction


def calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions(
        rule_wrapper: RuleWrapper,
        df_cached_predictions: pd.DataFrame,
        pylo_context: PyloContext,
        propensity_score_controller,
        verbose: bool = False
) -> None:

    if len(df_cached_predictions) != 0:
        rule_head: PyloAtom = rule_wrapper.rule.get_head()
        inverse_propensity_scores: pd.Series = get_inverse_propensity_score_per_prediction(
            df_predictions_to_weight=df_cached_predictions,
            rule_head=rule_head,
            pylo_context=pylo_context,
            propensity_score_controller=propensity_score_controller
        )
        if len(inverse_propensity_scores) != len(df_cached_predictions):
            raise Exception()

        if verbose:
            print("calculating inverse-e-weighted conf est")
        pu_conf_unbiased: float = calculate_inverse_propensity_weighted_confidence_from_df_cache(
            df_cached_predictions=df_cached_predictions,
            rule_head=rule_head,
            pylo_context=pylo_context,
            propensity_score_controller=propensity_score_controller,
            o_propensity_score_per_prediction=inverse_propensity_scores
        )
        # absolute_pu_e_confidence = relative_pu_conf * 3
        rule_wrapper.o_relative_pu_confidence_unbiased = pu_conf_unbiased

        if verbose:
            print("calculating inverse-e-weighted conf_PCA est (S->O)")
        pu_conf_pca_subject_to_object: float = calculate_inverse_propensity_weighted_pca_confidence_from_df_cache(
            df_cached_predictions=df_cached_predictions,
            rule_head=rule_head,
            pylo_context=pylo_context,
            predict_object_entity=True,
            propensity_score_controller=propensity_score_controller,
            verbose=verbose,
            o_propensity_score_per_prediction=inverse_propensity_scores
        )
        rule_wrapper.o_relative_pu_confidence_pca_subject_to_object = pu_conf_pca_subject_to_object
        if verbose:
            print("calculating inverse-e-weighted conf_PCA est (O->S)")
        pu_conf_pca_object_to_subject = calculate_inverse_propensity_weighted_pca_confidence_from_df_cache(
            df_cached_predictions=df_cached_predictions,
            rule_head=rule_head,
            pylo_context=pylo_context,
            predict_object_entity=False,
            propensity_score_controller=propensity_score_controller,
            verbose=verbose,
            o_propensity_score_per_prediction=inverse_propensity_scores
        )
        rule_wrapper.o_relative_pu_confidence_pca_object_to_subject = pu_conf_pca_object_to_subject
    else:
        pass  # ZERO predictions

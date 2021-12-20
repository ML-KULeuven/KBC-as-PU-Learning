from typing import Optional

import pandas as pd
from pylo.language.lp import Atom as PyloAtom, Context as PyloContext

from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_pu_confidence_unbiased_from_cached_predictions import \
    get_inverse_propensity_weighted_count_of_predictions


def calculate_pu_propensity_confidence_pca_from_df_cache(
        df_cached_predictions: pd.DataFrame,
        rule_head: PyloAtom,
        pylo_context: PyloContext,
        predict_object_entity: bool,
        propensity_score_controller,
        verbose: bool = False,
        o_propensity_score_per_prediction: Optional[pd.Series] = None
) -> Optional[float]:
    if len(df_cached_predictions) > 0:
        if predict_object_entity:
            pca_negatives_column_name = 'exists_lits_same_subject'
        else:
            pca_negatives_column_name = 'exists_lits_same_object'

        mask_known_positives = df_cached_predictions['is_supported']
        mask_pca_negatives = (~df_cached_predictions['is_supported']) & df_cached_predictions[pca_negatives_column_name]

        if o_propensity_score_per_prediction is None:
            df_known_positives: pd.DataFrame = df_cached_predictions[
                mask_known_positives
            ]
            weighted_known_positives: float = get_inverse_propensity_weighted_count_of_predictions(
                df_predictions_to_weight=df_known_positives,
                rule_head=rule_head,
                pylo_context=pylo_context,
                propensity_score_controller=propensity_score_controller
            )

            df_pca_negatives: pd.DataFrame = df_cached_predictions[
                mask_pca_negatives
            ]
            weighted_pca_negatives: float = get_inverse_propensity_weighted_count_of_predictions(
                df_predictions_to_weight=df_pca_negatives,
                rule_head=rule_head,
                pylo_context=pylo_context,
                propensity_score_controller=propensity_score_controller
            )
        else:
            weighted_known_positives: float = float(o_propensity_score_per_prediction[mask_known_positives].sum())
            weighted_pca_negatives: float = float(o_propensity_score_per_prediction[mask_pca_negatives].sum())
        weighted_denominator: float = weighted_known_positives + weighted_pca_negatives
        if weighted_denominator == 0:
            if verbose:
                print(f"Cannot compute unbiased PU conf,\n"
                      f"\t{weighted_known_positives} weighted known positives,"
                      f" {weighted_pca_negatives} weigthed PCA negatives")
            return None
        pu_confidence: float = weighted_known_positives / (weighted_known_positives + weighted_pca_negatives)
        return pu_confidence
    else:
        return None

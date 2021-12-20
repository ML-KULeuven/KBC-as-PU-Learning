from typing import Union

import pandas as pd

from kbc_pul.observed_data_generation.abstract_propensity_score_contoller import \
    AbstractPropensityScoreController

from pylo.language.lp import Atom as PyloAtom


class ConstantAddedNoisePropensityScoreController(AbstractPropensityScoreController):

    MINIMUM_PROP_SCORE: float = 0.01

    def __init__(self, constant_additive_noise: float,
                 true_propensity_score_controller: AbstractPropensityScoreController):
        self.constant_additive_noise: float = constant_additive_noise
        self.true_propensity_score_controller: AbstractPropensityScoreController = true_propensity_score_controller

    def get_propensity_score_of(self,
                                ground_literal: Union[pd.Series, PyloAtom]
                                ) -> float:
        true_prop_score: float = self.true_propensity_score_controller.get_propensity_score_of(
            ground_literal=ground_literal)
        noisy_prop_score: float = true_prop_score + self.constant_additive_noise
        if noisy_prop_score <= 0:
            noisy_prop_score = ConstantAddedNoisePropensityScoreController.MINIMUM_PROP_SCORE
        if noisy_prop_score > 1:
            noisy_prop_score = 1
        return noisy_prop_score

    def get_label_frequency_given_df_observed_literals(self,
                                                       df_observed_target_relation: pd.DataFrame
                                                       ) -> float:

        series_noisy_prop_scores: pd.Series = df_observed_target_relation.apply(
            func=lambda row: self.get_propensity_score_of(row),
            axis=1,
            result_type='reduce'
        )
        noisy_label_freq: float = float(series_noisy_prop_scores.mean())
        return noisy_label_freq




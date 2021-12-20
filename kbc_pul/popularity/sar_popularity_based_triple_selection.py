from random import Random
from typing import Optional

import pandas as pd

from kbc_pul.observed_data_generation.abstract_triple_selection import AbstractTripleSelector, \
    ObservedTargetRelationInfo
from kbc_pul.observed_data_generation.selection_mechanism_random_choosing import \
    make_random_observation_choice
from kbc_pul.popularity.abstract_popularity_based_propensity_score_controller import \
    AbstractPopularityBasedPropensityScoreController


class PopularityBasedTripleSelectorNonPCA(AbstractTripleSelector):

    def __init__(self,
                 popularity_based_propensity_score_controller: AbstractPopularityBasedPropensityScoreController,
                 verbose: bool = False
                 ):
        self.popularity_based_propensity_score_controller: AbstractPopularityBasedPropensityScoreController\
            = popularity_based_propensity_score_controller
        self.verbose: bool = verbose

        self._sum_of_propensity_scores: Optional[float] = None
        self.o_label_frequency_observed_relation: Optional = None

    def select_observed_target_relation(
            self,
            df_ground_truth_target_relation: pd.DataFrame,
            rng: Random
    ) -> ObservedTargetRelationInfo:

        self._sum_of_propensity_scores = 0

        # df_observed_target_relation: pd.DataFrame
        # mask_observed_rows: pd.Series
        observed_target_relation_info: ObservedTargetRelationInfo = super().select_observed_target_relation(
            df_ground_truth_target_relation=df_ground_truth_target_relation,
            rng=rng
        )
        n_observed_triples: int = len(observed_target_relation_info.df)
        if n_observed_triples == 0:
            self.o_label_frequency_observed_relation = 0
        else:
            self.o_label_frequency_observed_relation = self._sum_of_propensity_scores / n_observed_triples

        return observed_target_relation_info

    def should_select_triple(self, row: pd.Series, rng: Random) -> bool:
        subject_entity: str = row["Subject"]
        object_entity: str = row["Object"]
        propensity_score_to_use: float = self.popularity_based_propensity_score_controller.get_propensity_score_of(
            row
        )

        # Decide whether it is observed or not
        # - generate a uniform value between [0,1]
        # - accept if the value is smaller than the propensity score
        is_triple_observed: bool = make_random_observation_choice(random=rng,
                                                                  propensity_score=propensity_score_to_use)
        if is_triple_observed and self._sum_of_propensity_scores is not None:
            self._sum_of_propensity_scores += propensity_score_to_use

        if self.verbose:
            if is_triple_observed:
                decision_string = "SELECTED"
            else:
                decision_string = "NOT SELECTED"

            print(f"\tPCA decision: {subject_entity}, {object_entity} is {decision_string}")
        return is_triple_observed

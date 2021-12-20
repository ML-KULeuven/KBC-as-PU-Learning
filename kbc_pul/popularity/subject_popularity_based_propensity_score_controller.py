from typing import Union

import pandas as pd

from pylo.language.lp import Atom as PyloAtom, Constant as PyloConstant

from kbc_pul.observed_data_generation.abstract_propensity_score_contoller import \
    AbstractPropensityScoreController
from kbc_pul.popularity.entity_counting.count_to_normalized_popularity import \
    AbstractCountToNormalizedPopularityMapper

from kbc_pul.popularity.entity_counting.ent_count_in_other_rels_both_positions import \
    EntityCountInOtherRelationsBothPositions


class SubjectPopularityBasedPropensityScoreController(AbstractPropensityScoreController):

    def __init__(self,
                 entity_count_aggregator: EntityCountInOtherRelationsBothPositions,
                 count_to_normalized_popularity_mapper: AbstractCountToNormalizedPopularityMapper,
                 verbose: bool = False
                 ):

        self.entity_count_aggregator: EntityCountInOtherRelationsBothPositions = entity_count_aggregator
        self.count_to_norm_pop_mapper: AbstractCountToNormalizedPopularityMapper = count_to_normalized_popularity_mapper
        self.verbose: bool = verbose

    def get_propensity_score_of(self, ground_literal: Union[pd.Series, PyloAtom]) -> float:
        if isinstance(ground_literal, pd.Series):
            return self._get_group_score_pd_series(ground_literal)
        elif isinstance(ground_literal, PyloAtom):
            return self._get_group_score_pylo_atom(ground_literal)
        else:
            raise TypeError(f"Unexpected type: {type(ground_literal)}")

    def _get_group_score_pylo_atom(self, ground_literal: PyloAtom) -> float:
        first_argument: PyloConstant = ground_literal.get_arguments()[0]
        subject_entity_string: str = first_argument.name
        propensity_score: float = self.get_propensity_score_for_entity(subject_entity_string)
        return propensity_score

    def _get_group_score_pd_series(self, ground_literal: pd.Series) -> float:
        if not isinstance(ground_literal, pd.Series):
            raise TypeError(f"Expected pd.Series, but got {type(ground_literal)}")

        subject_entity: str = ground_literal["Subject"]
        propensity_score: float = self.get_propensity_score_for_entity(subject_entity)
        return propensity_score

    def get_propensity_score_for_entity(self, subject_entity: str) -> float:

        subject_entity_count: int = self.entity_count_aggregator.get_count(subject_entity)
        normalized_popularity: float = self.count_to_norm_pop_mapper.map_count_to_normalized_popularity(
            subject_entity_count)
        if self.verbose:
            print(f"{subject_entity}, count: {subject_entity_count}, norm pop: {normalized_popularity}")

        return normalized_popularity

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

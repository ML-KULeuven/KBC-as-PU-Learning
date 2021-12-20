from abc import ABC
from random import Random
from typing import Dict, Optional, NamedTuple

import pandas as pd

from kbc_pul.observed_data_generation.abstract_triple_selection import AbstractTripleSelector, \
    ObservedTargetRelationInfo
from kbc_pul.observed_data_generation.sar_two_subject_groups.subject_based_propensity_score_controller import \
    SetInclusionPropensityScoreController
from kbc_pul.observed_data_generation.selection_mechanism_random_choosing import \
    make_random_observation_choice


class SARSubjectGroupBasedTripleSelector(AbstractTripleSelector, ABC):

    def __init__(self,
                 entity_based_propensity_score_controller: SetInclusionPropensityScoreController,
                 verbose: bool = False,
                 ):
        self.entity_based_propensity_score_controller: SetInclusionPropensityScoreController \
            = entity_based_propensity_score_controller
        self.verbose: bool = verbose
        self.o_label_frequency_observed_relation: Optional[float] = None

        self._sum_of_propensity_scores: Optional[float] = None

    def label_frequency_of_observed_relation(self) -> float:
        if self.o_label_frequency_observed_relation is None:
            raise Exception("Label frequency of observed relation is None")
        else:
            return self.o_label_frequency_observed_relation

    def select_observed_target_relation(
            self,
            df_ground_truth_target_relation: pd.DataFrame,
            rng: Random
    ) -> ObservedTargetRelationInfo:

        self._sum_of_propensity_scores = 0

        df_observed_target_relation: pd.DataFrame
        mask_observed_rows: pd.Series
        df_observed_target_relation, mask_observed_rows = super().select_observed_target_relation(
            df_ground_truth_target_relation=df_ground_truth_target_relation,
            rng=rng
        )
        n_observed_triples: int = len(df_observed_target_relation)
        self.o_label_frequency_observed_relation = self._sum_of_propensity_scores / n_observed_triples

        return ObservedTargetRelationInfo(df_observed_target_relation, mask_observed_rows)


class SARSubjectGroupBasedTripleSelectorNonPCA(SARSubjectGroupBasedTripleSelector):
    def __init__(self, entity_based_propensity_score_controller: SetInclusionPropensityScoreController,
                 verbose: bool = False):
        super().__init__(entity_based_propensity_score_controller=entity_based_propensity_score_controller,
                         verbose=verbose)

    def should_select_triple(self, row: pd.Series, rng: Random) -> bool:
        subject_entity: str = row["Subject"]
        object_entity: str = row["Object"]
        propensity_score_to_use: float = self.entity_based_propensity_score_controller.get_propensity_score_for_entity(
            subject_entity
        )

        # Decide whether it is observed or not
        # - generate a uniform value between [0,1]
        # - accept if the value is smaller than the propensity score
        is_triple_observed: bool = make_random_observation_choice(random=rng,
                                                                  propensity_score=propensity_score_to_use)
        if is_triple_observed and self._sum_of_propensity_scores is not None:
            self._sum_of_propensity_scores += propensity_score_to_use

        if self.verbose:
            print(f"\tPCA decision: {subject_entity}, {object_entity} is SELECTED")
        return is_triple_observed


class SARSubjectGroupBasedTripleSelectorPCA(SARSubjectGroupBasedTripleSelector):

    def __init__(self, entity_based_propensity_score_controller: SetInclusionPropensityScoreController,
                 verbose: bool = False):
        super().__init__(entity_based_propensity_score_controller=entity_based_propensity_score_controller,
                         verbose=verbose)
        self.map_subject_entity_to_pca_choice: Dict[str, bool] = dict()
        self.n_pairs_counter: int = 0

    def should_select_triple(self, row: pd.Series, rng: Random) -> bool:
        subject_entity: str = row["Subject"]
        object_entity: str = row["Object"]

        propensity_score_to_use: float = self.entity_based_propensity_score_controller.get_propensity_score_for_entity(
            subject_entity
        )

        # CHECK if PCA observation decision has already been made for the current subject
        o_pca_is_observed: Optional[bool] = self.map_subject_entity_to_pca_choice.get(
            subject_entity, None)

        if self.verbose:
            print(f"Current row: {subject_entity} {object_entity}")

        if o_pca_is_observed is None:
            if self.verbose:
                print("\tpair has no PCA decision")
            # Decision must first be made
            # Decide whether it is observed or not
            # - generate a uniform value between [0,1]
            # - accept if the value is smaller than the propensity score
            o_pca_is_observed: bool = make_random_observation_choice(
                random=rng, propensity_score=propensity_score_to_use)
            if o_pca_is_observed:
                self.n_pairs_counter += 1

            self.map_subject_entity_to_pca_choice[subject_entity] = o_pca_is_observed

            if self.verbose:
                print(f"\tprop score: {propensity_score_to_use}")
                if o_pca_is_observed:
                    print(f"\tPCA decision: {subject_entity} is SELECTED")
                else:
                    print(f"\tPCA decision: {subject_entity} is NOT SELECTED")
        else:
            if self.verbose:
                print("\tpair already has PCA decision")
                print(f"\tPCA decision: {subject_entity} is SELECTED")

        if o_pca_is_observed and self._sum_of_propensity_scores is not None:
            self._sum_of_propensity_scores += propensity_score_to_use

        return o_pca_is_observed


def get_subject_group_based_triple_selector(
        use_pca_version: bool,
        entity_based_propensity_score_controller: SetInclusionPropensityScoreController,
        verbose: bool = False
) -> SARSubjectGroupBasedTripleSelector:
    if use_pca_version:
        return SARSubjectGroupBasedTripleSelectorPCA(
            entity_based_propensity_score_controller=entity_based_propensity_score_controller,
            verbose=verbose
        )
    else:
        return SARSubjectGroupBasedTripleSelectorNonPCA(
            entity_based_propensity_score_controller=entity_based_propensity_score_controller,
            verbose=verbose
        )

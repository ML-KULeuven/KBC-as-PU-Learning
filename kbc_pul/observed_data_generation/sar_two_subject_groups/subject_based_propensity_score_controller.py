from abc import abstractmethod, ABC
from typing import Set, Union

import pandas as pd
from pylo.language.lp import Atom as PyloAtom, Constant as PyloConstant

from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.observed_data_generation.abstract_propensity_score_contoller import \
    AbstractPropensityScoreController
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


class EntityBasedPropensityScoreController(AbstractPropensityScoreController, ABC):
    @abstractmethod
    def get_propensity_score_for_entity(self, entity_string: str) -> float:
        pass


class SetInclusionPropensityScoreController(EntityBasedPropensityScoreController):
    def __init__(self,
                 subject_entity_set: Set[str],
                 propensity_score_if_in_set: float,
                 propensity_score_if_not_in_set: float
                 ):
        self._subject_entity_set: Set[str] = subject_entity_set
        self._propensity_score_if_in_set: float = propensity_score_if_in_set
        self._propensity_score_if_not_in_set: float = propensity_score_if_not_in_set

    def get_subject_entity_set(self) -> Set[str]:
        return self._subject_entity_set

    def get_mask_of_literals_in_set(self, df_observed_relation: pd.DataFrame) -> pd.Series:
        PandasKnowledgeBaseWrapper.check_if_df_in_pandas_kb_format(df_observed_relation)

        mask_observed_literals_in_set: pd.Series = df_observed_relation.apply(
            func=lambda row: row["Subject"] in self._subject_entity_set,
            axis=1,
            result_type='reduce'
        )
        return mask_observed_literals_in_set

    def get_label_frequency_given_mask_observed_literals_in_set(self,
                                                                mask_observed_literals_in_set: pd.Series
                                                                ) -> float:
        n_observed_literals: int = len(mask_observed_literals_in_set)
        n_observed_literals_in_set: int = int(mask_observed_literals_in_set.sum())
        n_observed_literals_not_in_set: int = n_observed_literals - n_observed_literals_in_set

        observed_label_frequency: float = (
            self._propensity_score_if_in_set * n_observed_literals_in_set
            +
            self._propensity_score_if_not_in_set * n_observed_literals_not_in_set
        ) / n_observed_literals
        return observed_label_frequency

    def get_propensity_score_for_entity(self, entity_string: str) -> float:
        if entity_string in self._subject_entity_set:
            return self._propensity_score_if_in_set
        else:
            return self._propensity_score_if_not_in_set

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


def get_subject_entities_of_filter_relation(
        df_ground_truth: pd.DataFrame,
        filter_relation: str
) -> Set[str]:
    subject_entity_set: Set[str] = set(
        df_ground_truth[df_ground_truth["Rel"] == filter_relation]["Subject"].unique()
    )
    return subject_entity_set


def build_set_inclusion_propensity_score_controller_from(
        df_ground_truth: pd.DataFrame,
        filter_relation: str,
        prop_scores_two_groups: PropScoresTwoSARGroups,
        verbose: bool = False
) -> SetInclusionPropensityScoreController:
    subjects_of_filter_relation_set: Set[str] = get_subject_entities_of_filter_relation(
        df_ground_truth=df_ground_truth,
        filter_relation=filter_relation
    )
    if verbose:
        print(f"\t{len(subjects_of_filter_relation_set)} {filter_relation} (filter) subject entities")

    return SetInclusionPropensityScoreController(
        subject_entity_set=subjects_of_filter_relation_set,
        propensity_score_if_in_set=prop_scores_two_groups.in_filter,
        propensity_score_if_not_in_set=prop_scores_two_groups.other
    )

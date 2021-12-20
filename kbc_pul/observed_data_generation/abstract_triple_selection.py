from abc import ABC, abstractmethod
from collections import namedtuple
from random import Random
from typing import Tuple, NamedTuple

import pandas as pd


class ObservedTargetRelationInfo(NamedTuple):
    df: pd.DataFrame
    mask: pd.Series


class AbstractTripleSelector(ABC):

    @abstractmethod
    def should_select_triple(self, row: pd.Series, rng: Random) -> bool:
        pass

    def select_observed_target_relation(
            self,
            df_ground_truth_target_relation: pd.DataFrame,
            rng: Random
    ) -> ObservedTargetRelationInfo:
        # Decide whether a row is observed or not
        # - generate a uniform value between [0,1]
        # - accept if the value is smaller than the propensity score
        mask_observed_rows: pd.Series = df_ground_truth_target_relation.apply(
            lambda row: self.should_select_triple(row, rng),
            axis=1,
            result_type='reduce'
        )
        df_observed_target_relation: pd.DataFrame = df_ground_truth_target_relation[mask_observed_rows]
        return ObservedTargetRelationInfo(df_observed_target_relation, mask_observed_rows)

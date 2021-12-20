from random import Random
from typing import Tuple

import pandas as pd

from kbc_pul.observed_data_generation.abstract_triple_selection import AbstractTripleSelector
from kbc_pul.observed_data_generation.selection_mechanism_random_choosing import \
    make_random_observation_choice


class SCARTripleSelector(AbstractTripleSelector):
    def __init__(self,
                 constant_label_frequency: float,
                 verbose: bool = False):
        self.constant_label_frequency: float = constant_label_frequency
        self.verbose: bool = verbose

    def should_select_triple(self, row: pd.Series, rng: Random) -> bool:
        subject_entity: str = row["Subject"]
        object_entity: str = row["Object"]
        is_triple_observed: bool = make_random_observation_choice(random=rng,
                                                                  propensity_score=self.constant_label_frequency)
        if self.verbose:
            print(f"\tSCAR : {subject_entity}, {object_entity} is SELECTED")
        return is_triple_observed

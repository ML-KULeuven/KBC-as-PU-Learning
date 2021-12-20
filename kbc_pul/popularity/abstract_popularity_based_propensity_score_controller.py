from typing import Union

import pandas as pd
from pylo.language.commons import Atom as PyloAtom

from kbc_pul.observed_data_generation.abstract_propensity_score_contoller import \
    AbstractPropensityScoreController


class AbstractPopularityBasedPropensityScoreController(AbstractPropensityScoreController):
    def get_propensity_score_of(self, ground_literal: Union[pd.Series, PyloAtom]) -> float:
        raise NotImplementedError()

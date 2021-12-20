from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from pylo.language.lp import Atom as PyloAtom


class AbstractPropensityScoreController(ABC):

    @abstractmethod
    def get_propensity_score_of(self, ground_literal: Union[pd.Series, PyloAtom]) -> float:
        pass

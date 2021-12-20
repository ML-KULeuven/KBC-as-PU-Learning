from abc import abstractmethod
from functools import partial
from typing import Callable

from kbc_pul.popularity.logistic_functions import logistic_popularity_function


class AbstractCountToNormalizedPopularityMapper:
    """
    Maps a value in R to [0,1]
    """

    @classmethod
    def is_value_normalized(cls, value) -> bool:
        return 0 <= value <= 1

    @abstractmethod
    def map_count_to_normalized_popularity(self, count: int) -> float:
        pass


class LogisticCountToNormalizedPopularityMapper(AbstractCountToNormalizedPopularityMapper):

    MINIMUM_POPENSITY_SCORE_VALUE = 0.01

    def __init__(self, log_growth_rate: float):
        self.function: Callable = partial(
            logistic_popularity_function, log_growth_rate=log_growth_rate
        )

    def map_count_to_normalized_popularity(self, count: int) -> float:

        value = self.function(count)
        if value < LogisticCountToNormalizedPopularityMapper.MINIMUM_POPENSITY_SCORE_VALUE:
            value = LogisticCountToNormalizedPopularityMapper.MINIMUM_POPENSITY_SCORE_VALUE

        return value
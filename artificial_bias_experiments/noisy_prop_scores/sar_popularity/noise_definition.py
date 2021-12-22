from abc import abstractmethod, ABC, abstractclassmethod
from enum import Enum
from typing import List

from kbc_pul.observed_data_generation.abstract_propensity_score_contoller import \
    AbstractPropensityScoreController
from kbc_pul.popularity.constant_noise_subject_pop_based_propensity_controller import \
    ConstantAddedNoisePropensityScoreController
from kbc_pul.popularity.entity_counting.count_to_normalized_popularity import \
    LogisticCountToNormalizedPopularityMapper
from kbc_pul.popularity.subject_popularity_based_propensity_score_controller import \
    SubjectPopularityBasedPropensityScoreController


class NoiseTypeSARPopularity(Enum):
    CONSTANT_ADDED_TO_PROP_SCORE = "constant_noise_added_to_prop_score"
    FRACTION_OF_LOG_GROWTH_RATE = "noise_as_fraction_of_log_growth_rate"

    def get_noise_definition(self) -> 'NoiseDefinition':
        if self is NoiseTypeSARPopularity.CONSTANT_ADDED_TO_PROP_SCORE:
            return AdditiveNoise()
        elif self is NoiseTypeSARPopularity.FRACTION_OF_LOG_GROWTH_RATE:
            return NoisyLogGrowthRate()
        else:
            raise Exception()


class NoiseDefinition(ABC):

    @classmethod
    @abstractmethod
    def get_noisy_propensity_score_controller(cls,
                                              noise_level: float,
                                              true_log_growth_rate: float,
                                              true_propensity_score_controller: AbstractPropensityScoreController
                                              ):
        pass


class AdditiveNoise(NoiseDefinition):
    # def __init__(self, additive_noise_levels: List[float]):
    #     self.additive_noise_levels: List[float] = additive_noise_levels
    #
    # def get_noise_levels(self) -> List[float]:
    #     return self.additive_noise_levels

    @classmethod
    def get_noisy_propensity_score_controller(cls,
                                              noise_level: float,
                                              true_log_growth_rate: float,
                                              true_propensity_score_controller: AbstractPropensityScoreController
                                              ):
        noisy_propensity_score_controller = ConstantAddedNoisePropensityScoreController(
            constant_additive_noise=noise_level,
            true_propensity_score_controller=true_propensity_score_controller
        )
        return noisy_propensity_score_controller


class NoisyLogGrowthRate(NoiseDefinition):
    # def __init__(self, ):
    #     pass
    @classmethod
    def get_noisy_propensity_score_controller(cls,
                                              noise_level: float,
                                              true_log_growth_rate: float,
                                              true_propensity_score_controller: SubjectPopularityBasedPropensityScoreController
                                              ):
        noisy_log_growth_rate: float = true_log_growth_rate * noise_level

        noisy_count_to_normalized_popularity_mapper: LogisticCountToNormalizedPopularityMapper = LogisticCountToNormalizedPopularityMapper(
            log_growth_rate=noisy_log_growth_rate
        )
        noisy_propensity_score_controller = SubjectPopularityBasedPropensityScoreController(
            entity_count_aggregator=true_propensity_score_controller.entity_count_aggregator,
            count_to_normalized_popularity_mapper=noisy_count_to_normalized_popularity_mapper
        )
        return noisy_propensity_score_controller

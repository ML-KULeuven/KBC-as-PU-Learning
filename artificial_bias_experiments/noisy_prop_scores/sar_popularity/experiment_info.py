from typing import NamedTuple, List

from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noise_definition import \
    NoiseTypeSARPopularity
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


class NoisyPropScoresSARPopularityExperimentInfo(NamedTuple):
    dataset_name: str
    target_relation: str
    is_pca_version: bool
    log_growth_rate: float
    noise_type: NoiseTypeSARPopularity
    noise_levels: List[float]



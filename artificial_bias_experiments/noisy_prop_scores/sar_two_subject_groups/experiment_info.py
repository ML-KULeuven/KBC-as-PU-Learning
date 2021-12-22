from typing import List, NamedTuple

from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


class NoisyPropScoresSARExperimentInfo(NamedTuple):
    dataset_name: str
    target_relation: str
    filter_relation: str
    true_prop_scores: PropScoresTwoSARGroups
    noisy_prop_score_in_filter: float
    noisy_prop_score_not_in_filter_list: List[float]
    is_pca_version: bool

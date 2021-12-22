from typing import NamedTuple

from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


class KnownPropScoresSARExperimentInfo(NamedTuple):
    dataset_name: str
    target_relation: str
    filter_relation: str
    true_prop_scores: PropScoresTwoSARGroups
    is_pca_version: bool


class TargetFilterOverlapSettings(NamedTuple):
    fraction_lower_bound: float
    fraction_upper_bound: float
    intersection_absolute_lower_bound: int

    def is_strictly_between_fraction_bounds(self, fraction_to_check: float) -> bool:
        return self.fraction_lower_bound < fraction_to_check < self.fraction_upper_bound

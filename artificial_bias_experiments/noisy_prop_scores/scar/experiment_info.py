from typing import List, NamedTuple


class NoisyPropScoresSCARExperimentInfo(NamedTuple):
    dataset_name: str
    target_relation: str
    true_label_frequency: float
    available_label_frequency_list: List[float]
    is_pca_version: bool

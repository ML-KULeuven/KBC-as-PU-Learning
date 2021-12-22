from typing import NamedTuple


class KnownPropScoresSCARExperimentInfo(NamedTuple):
    dataset_name: str
    target_relation: str
    true_label_frequency: float
    is_pca_version: bool

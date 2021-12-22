import csv
import gzip
import json
from typing import Dict, Optional, Tuple, List

import pandas as pd

from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups

ConfIDStr = str


class RuleWrapperNoiseLevelToPuMetricsMap:
    """
    GOAL:
        For a single rule, we want to compute the PU-based confidence estimators
            using different noisy propensity scores.

    This object transforms the rule


    """

    def __init__(self,
                 rule_str: str,
                 random_trial_index: int,
                 log_growth_rate: float
                 ):
        self.rule_str: str = rule_str
        self.random_trial_index: int = random_trial_index
        self.log_growth_rate: float = log_growth_rate

        self._noise_level_to_pu_metrics_map: Dict[float,
                                                  Dict[ConfIDStr, float]] = dict()

    def add_pu_metrics_for_noise_level(self,
                                       noise_level: float,
                                       o_relative_pu_confidence_unbiased: Optional[float],
                                       o_relative_pu_confidence_pca_object_to_subject: Optional[float],
                                       o_relative_pu_confidence_pca_subject_to_object: Optional[float],
                                       o_inverse_c_weighted_std_confidence: Optional[float]
                                       ) -> None:
        self._noise_level_to_pu_metrics_map[noise_level] = dict(
            o_relative_pu_confidence_unbiased=o_relative_pu_confidence_unbiased,
            o_relative_pu_confidence_pca_subject_to_object=o_relative_pu_confidence_pca_subject_to_object,
            o_relative_pu_confidence_pca_object_to_subject=o_relative_pu_confidence_pca_object_to_subject,
            o_inverse_c_weighted_std_confidence=o_inverse_c_weighted_std_confidence
        )

    @staticmethod
    def get_column_names():
        return [
                   "Rule",
                   "random_trial_index",
                    "log_growth_rate",
                    "noise_level",
                   ConfidenceEnum.IPW_CONF.get_name(),
                   ConfidenceEnum.IPW_PCA_CONF_S_TO_O.get_name(),
                   ConfidenceEnum.IPW_PCA_CONF_O_TO_S.get_name(),

                   ConfidenceEnum.ICW_CONF.get_name(),
               ]

    def to_tsv(self,
               filename_noise_level_to_pu_metrics_map: str,
               ) -> None:
        with gzip.open(filename_noise_level_to_pu_metrics_map, "wt") as output_file:
            csv_writer = csv.writer(output_file, delimiter="\t")

            noise_level: float
            metric_dict: Dict[ConfIDStr, float]
            for noise_level, metric_dict in self._noise_level_to_pu_metrics_map.items():
                csv_writer.writerow(
                    [self.rule_str,
                     self.random_trial_index,

                     self.log_growth_rate,

                     noise_level,

                     metric_dict["o_relative_pu_confidence_unbiased"],
                     metric_dict["o_relative_pu_confidence_pca_subject_to_object"],
                     metric_dict["o_relative_pu_confidence_pca_subject_to_object"],

                     metric_dict["o_inverse_c_weighted_std_confidence"],
                     ]
                )

    @classmethod
    def read_csv(cls, filename_noisy_prop_scores_to_pu_metrics_map: str) -> pd.DataFrame:
        df_pu_metrics = pd.read_csv(
            filename_noisy_prop_scores_to_pu_metrics_map,
            sep="\t",
            names=cls.get_column_names()
        )
        return df_pu_metrics

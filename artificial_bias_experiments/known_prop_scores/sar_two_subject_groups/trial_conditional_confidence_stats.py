import os
from typing import List, Union, Tuple

import pandas as pd

from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.conditioned_true_confidence_tuple import \
    ConditionedTrueConfidenceTuple
from kbc_pul.data_structures.rule_wrapper import RuleWrapper
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


class TrialConditionalTrueConfidencesManager:
    """
    A class to temporarily store the TRUE CONFIDENCE of the rules over multiple trials,
        which can then be output to a single file.
    Note: for non-recursive rules, the true confidence for each trial should be the same in principle.
    The True confidences are:
        * the default confidence
        * the positive-pair confidence in both directions.


    """


    def __init__(self, experiment_info, random_trial_index: int):
        self.data_conditional_confidence_rule_stats: List[
            List[Union[str, float, int]]
        ] = []
        self.one_interesting_rule: bool = False

        self.experiment_info = experiment_info
        self.random_trial_index: int = random_trial_index

    @staticmethod
    def get_column_names() -> List[str]:
        column_names: List[str] = (
                [
                    "Rule",
                    "random_trial_index",
                    "target_relation",
                    "filter_relation",
                ]
                + PropScoresTwoSARGroups.get_column_names("true_")

                + ["true_conf"]
                + ConditionedTrueConfidenceTuple.get_confidence_column_names("true_")

                + ["true_pos_pair_conf_s_to_o"]
                + ConditionedTrueConfidenceTuple.get_confidence_column_names("true_pos_pair_s_to_o_")

                + ["true_pos_pair_conf_o_to_s"]
                + ConditionedTrueConfidenceTuple.get_confidence_column_names("true_pos_pair_o_to_s_")

                + ConditionedTrueConfidenceTuple.get_n_predictions_column_names()
                + ['n_predictions']
        )
        return column_names

    def add_rule_wrapper_info(self,
                              rule_wrapper: RuleWrapper,
                              conditioned_true_conf_tuple: ConditionedTrueConfidenceTuple,
                              conditioned_true_pos_pair_tuples: Tuple[
                                  ConditionedTrueConfidenceTuple, ConditionedTrueConfidenceTuple]
                              ) -> None:
        total_n_predictions: int = (
                conditioned_true_conf_tuple.n_predictions_in_filter  # AttributeError: 'NoneType' object has no attribute 'n_predictions_in_filter'
                + conditioned_true_conf_tuple.n_predictions_not_in_filter
        )

        s_t_o_tuple = conditioned_true_pos_pair_tuples[0]
        o_to_s_tuple = conditioned_true_pos_pair_tuples[1]

        data_row_conditional_conf_rule_stats: List[Union[str, float]] = (
                [
                    str(rule_wrapper.rule),
                    self.random_trial_index,
                    self.experiment_info.target_relation,
                    self.experiment_info.filter_relation,
                ]
                + self.experiment_info.true_prop_scores.to_rows()

                + [
                    rule_wrapper.o_true_confidence,
                    conditioned_true_conf_tuple.true_conf_on_predictions_in_filter,
                    conditioned_true_conf_tuple.true_conf_on_predictions_not_in_filter
                ]
                + [
                    rule_wrapper.o_true_pca_confidence_subject_to_object,
                    s_t_o_tuple.true_conf_on_predictions_in_filter,
                    s_t_o_tuple.true_conf_on_predictions_not_in_filter
                ]
                + [
                    rule_wrapper.o_true_pca_confidence_object_to_subject,
                    o_to_s_tuple.true_conf_on_predictions_in_filter,
                    o_to_s_tuple.true_conf_on_predictions_not_in_filter
                ]
                + [
                    conditioned_true_conf_tuple.n_predictions_in_filter,
                    conditioned_true_conf_tuple.n_predictions_not_in_filter,
                    total_n_predictions
                ]
        )

        self.data_conditional_confidence_rule_stats.append(data_row_conditional_conf_rule_stats)
        self.one_interesting_rule = self.one_interesting_rule or conditioned_true_conf_tuple.is_interesting()

    def make_file_indicating_at_least_one_rule_was_interesting(self,
                                                               experiment_dir: str) -> None:
        filename_at_least_one_interesting_rule_exists: str = os.path.join(
            experiment_dir,
            f"at_least_one_interesting_rule_{self.experiment_info.target_relation}"
            f"_{self.experiment_info.filter_relation}"
            f"_s_prop{self.experiment_info.true_prop_scores.in_filter}"
            f"_ns_prop{self.experiment_info.true_prop_scores.other}"
            f"_trial{self.random_trial_index}.txt"
        )
        with open(filename_at_least_one_interesting_rule_exists, 'w') as fp:
            pass

    @staticmethod
    def get_filename(experiment_dir: str,
                     experiment_info,
                     random_trial_index: int
                     ) -> str:
        filename_conditional_confidence_rule_stats: str = os.path.join(
            experiment_dir,
            f"conditional_conf_rule_stats_{experiment_info.target_relation}_{experiment_info.filter_relation}"
            f"_s_prop{experiment_info.true_prop_scores.in_filter}"
            f"_ns_prop{experiment_info.true_prop_scores.other}_trial{random_trial_index}.csv"
        )
        return filename_conditional_confidence_rule_stats

    def to_csv(self, experiment_dir: str) -> None:
        df_stats = pd.DataFrame(
            data=self.data_conditional_confidence_rule_stats,
            columns=self.get_column_names()
        )
        filename_conditional_confidence_rule_stats: str = self.get_filename(
            experiment_dir=experiment_dir,
            experiment_info=self.experiment_info,
            random_trial_index=self.random_trial_index
        )

        df_stats.to_csv(
            filename_conditional_confidence_rule_stats,
            header=True,
            sep="\t",
            index=False
        )

    @staticmethod
    def read_csv(filename: str) -> pd.DataFrame:
        return pd.read_csv(
            filename,
            sep="\t",
        )

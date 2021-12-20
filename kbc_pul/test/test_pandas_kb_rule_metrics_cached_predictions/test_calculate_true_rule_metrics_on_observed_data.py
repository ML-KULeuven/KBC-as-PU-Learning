import unittest
from typing import List, Dict, Tuple, Set, Optional

import pandas as pd

from artificial_bias_experiments.evaluation.ground_truth_utils import get_true_entity_sets_as_string_sets, \
    TrueEntitySetsTuple
from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.data_structures.rule_wrapper import RuleWrapper

from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_true_confidence_on_observed_data_from_cached_predictions import \
    calculate_true_confidence_metrics_from_df_cached_predictions

from kbc_pul.test.rule_wrapper_testing_utils import get_rule_wrapper_from_str_repr


class TestinTrueConfidenceMetricsOnObservedData(unittest.TestCase):
    def setUp(self):
        observed_data: List[List[str]] = [
            ["'adam'", "livesin", "'paris'"],
            ["'adam'", "livesin", "'rome'"],
            ["'bob'", "livesin", "'zurich'"],

            ["'adam'", "wasbornin", "'paris'"],
            ["'carl'", "wasbornin", "'rome'"],
            ["'dennis'", "wasbornin", "'zurich'"]  # added to have a different PCA conf in both directions
        ]
        columns = ["Subject", "Rel", "Object"]

        true_data = observed_data + [
            ["'adam'", "wasbornin", "'rome'"]
        ]

        self.df_observed: pd.DataFrame = pd.DataFrame(data=observed_data, columns=columns)

        self.df_ground_truth: pd.DataFrame = pd.DataFrame(data=true_data, columns=columns)

        rule_string: str = "wasbornin(X,Y) :- livesin(X,Y)"
        self.rule_wrapper: RuleWrapper = get_rule_wrapper_from_str_repr(rule_string)

        self.correct_true_std_confidence: float = 2 / 3
        self.correct_true_pca_confidence_subject_to_object: float = 1
        self.correct_true_pca_confidence_object_to_subject: float = 2/3

    def test_true_confidence_metrics_on_observed_data(self):

        kb_wrapper: PandasKnowledgeBaseWrapper = PandasKnowledgeBaseWrapper.create_from_full_data(
            df_full_data=self.df_observed
        )
        df_prediction_cache: Optional[pd.DataFrame] = kb_wrapper.calculate_prediction_cache_for_rule(rule=self.rule_wrapper.rule)

        df_ground_truth_target_relation: pd.DataFrame = self.df_ground_truth[
            self.df_ground_truth["Rel"] == self.rule_wrapper.rule.get_head().get_predicate().get_name()
        ]

        true_entity_sets: TrueEntitySetsTuple = get_true_entity_sets_as_string_sets(
            df_ground_truth_target_relation
        )
        calculate_true_confidence_metrics_from_df_cached_predictions(
            rule_wrapper=self.rule_wrapper,
            df_cached_predictions=df_prediction_cache,
            df_ground_truth_target_relation=df_ground_truth_target_relation,
            true_entity_sets=true_entity_sets
        )

        self.assertAlmostEqual(self.rule_wrapper.o_true_confidence,
                               self.correct_true_std_confidence, places=4)
        self.assertAlmostEqual(self.rule_wrapper.o_true_pca_confidence_subject_to_object,
                               self.correct_true_pca_confidence_subject_to_object, places=4)
        self.assertAlmostEqual(self.rule_wrapper.o_true_pca_confidence_object_to_subject,
                               self.correct_true_pca_confidence_object_to_subject, places=4)

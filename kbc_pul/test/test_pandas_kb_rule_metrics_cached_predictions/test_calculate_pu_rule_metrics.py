import unittest
from typing import List, Optional

import pandas as pd

from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.data_structures.rule_wrapper import RuleWrapper
from kbc_pul.rule_metrics.prolog_kb_rule_metrics.rule_pu_metrics import calculate_rule_pu_metrics

from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_pu_metrics_from_cached_predictions import \
    calculate_rule_pu_metrics_from_df_cached_predictions
from kbc_pul.test.rule_wrapper_testing_utils import get_rule_wrapper_from_str_repr

from pylo.language.lp import (Clause as PyloClause, global_context as pylo_global_context)



class TestingRegularConfidenceMetrics(unittest.TestCase):
    def setUp(self):
        data: List[List[str]] = [
            ["'adam'", "livesin", "'paris'"],
            ["'adam'", "livesin", "'rome'"],
            ["'bob'", "livesin", "'zurich'"],

            ["'adam'", "wasbornin", "'paris'"],
            ["'carl'", "wasbornin", "'rome'"],

            ["'dennis'", "wasbornin", "'zurich'"]  # added to have a different PCA conf in both directions
        ]
        columns = ["Subject", "Rel", "Object"]
        self.df: pd.DataFrame = pd.DataFrame(data=data, columns=columns)

        rule_string: str = "wasbornin(X,Y) :- livesin(X,Y)"
        self.rule_wrapper: RuleWrapper = get_rule_wrapper_from_str_repr(rule_string)

        self.correct_relative_pu_confidence_uniased: float = 1 / 3
        self.correct_pca_pu_confidence_subject_to_object: float = 0.5
        self.correct_pca_pu_confidence_object_to_subject: float = 1/3

    def test_regular_confidence_metrics(self):
        kb_wrapper: PandasKnowledgeBaseWrapper = PandasKnowledgeBaseWrapper.create_from_full_data(
            df_full_data=self.df
        )
        df_prediction_cache: Optional[pd.DataFrame] = kb_wrapper.calculate_prediction_cache_for_rule(
            rule=self.rule_wrapper.rule)

        calculate_rule_pu_metrics_from_df_cached_predictions(
            rule_wrapper=self.rule_wrapper,
            df_cached_predictions=df_prediction_cache,
            pylo_context=pylo_global_context,
            propensity_score_controller=true_set_inclusion_prop_score_controller,
            verbose=False
        )


        # self.assertAlmostEqual(self.rule_wrapper.o_relative_pu_confidence_unbiased,
        #                        self.correct_relative_pu_confidence_uniased, places=4
        #                        )
        # self.assertAlmostEqual(self.rule_wrapper.o_relative_pu_confidence_pca_subject_to_object,
        #                        self.correct_pca_pu_confidence_subject_to_object, places=4)
        # self.assertAlmostEqual(self.rule_wrapper.o_relative_pu_confidence_pca_object_to_subject,
        #                        self.correct_pca_pu_confidence_object_to_subject, places=4)

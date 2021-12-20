# import unittest
# from typing import List, Dict, Tuple, Set
#
# import pandas as pd
#
# # from kbc_pul.data_structures.prolog_kb import KnowledgeBaseWrapper
# # from kbc_pul.test.kb_testing_utils import init_knowledge_base_from_dataframe
#
# from pylo.engines.prolog.prologsolver import Prolog as PyloProlog
#
# from pylo.language.lp import (
#     Context as PyloContext, Atom as PyloAtom, Predicate as PyloPredicate,
#     Variable as PyloVariable, Constant as PyloConstant
# )
#
#
# class TestKnowledgeBaseWrapperWithAmiePaperExample(unittest.TestCase):
#     def setUp(self):
#         data: List[List[str]] = [
#             ["'adam'", "livesin", "'paris'"],
#             ["'adam'", "livesin", "'rome'"],
#             ["'bob'", "livesin", "'zurich'"],
#
#             ["'adam'", "wasbornin", "'paris'"],
#             ["'carl'", "wasbornin", "'rome'"],
#         ]
#         columns = ["Subject", "Rel", "Object"]
#
#         self.expected_livesin_results: Set[Tuple[str, str]] = {
#             ("'adam'", "'paris'"),
#             ("'adam'", "'rome'"),
#             ("'bob'", "'zurich'"),
#         }
#         self.expected_wasbornin_results: Set[Tuple[str, str]] = {
#             ("'adam'", "'paris'"),
#             ("'carl'", "'rome'"),
#         }
#
#         self.df: pd.DataFrame = pd.DataFrame(data=data, columns=columns)
#
#     def test_kb_correctly_initialized(self):
#         kb_wrapper: KnowledgeBaseWrapper = init_knowledge_base_from_dataframe(df_to_consult=self.df)
#
#         pylo_context: PyloContext = kb_wrapper.pylo_global_context
#         prolog_engine: PyloProlog = kb_wrapper.prolog_engine
#
#         pylo_variable_x = pylo_context.variable("X")
#         pylo_variable_y = pylo_context.variable("Y")
#
#         test_query_livesin_predicate: PyloPredicate = pylo_context.predicate("livesin", arity=2)
#         test_query_livesin_atom: PyloAtom = test_query_livesin_predicate(pylo_variable_x, pylo_variable_y)
#
#         test_query_wasbornin_predicate: PyloPredicate = pylo_context.predicate("wasbornin", arity=2)
#         test_query_wasbornin_atom: PyloAtom = test_query_wasbornin_predicate(pylo_variable_x, pylo_variable_y)
#
#         # testing livesin
#         results_test_query_livesin: List[
#             Dict[PyloVariable, PyloConstant]
#         ] = prolog_engine.query(test_query_livesin_atom)
#         self.assertEqual(len(results_test_query_livesin),
#                          len(self.expected_livesin_results)
#                          )
#
#         single_query_result: Dict[PyloVariable, PyloConstant]
#         for single_query_result in results_test_query_livesin:
#             const_bound_to_x: PyloConstant = single_query_result[pylo_variable_x]
#             const_bound_to_y: PyloConstant = single_query_result[pylo_variable_y]
#
#             result_string_tuple: Tuple[str, str] = (const_bound_to_x.name, const_bound_to_y.name)
#             self.assertTrue(
#                 result_string_tuple in self.expected_livesin_results
#             )
#
#         # testing wasbornin
#         results_test_query_wasbornin: List[
#             Dict[PyloVariable, PyloConstant]
#         ] = prolog_engine.query(test_query_wasbornin_atom)
#         self.assertEqual(len(results_test_query_wasbornin),
#                          len(self.expected_wasbornin_results)
#                          )
#
#         single_query_result: Dict[PyloVariable, PyloConstant]
#         for single_query_result in results_test_query_wasbornin:
#             const_bound_to_x: PyloConstant = single_query_result[pylo_variable_x]
#             const_bound_to_y: PyloConstant = single_query_result[pylo_variable_y]
#
#             result_string_tuple: Tuple[str, str] = (const_bound_to_x.name, const_bound_to_y.name)
#             self.assertTrue(
#                 result_string_tuple in self.expected_wasbornin_results
#             )
#
#
# if __name__ == '__main__':
#     unittest.main()

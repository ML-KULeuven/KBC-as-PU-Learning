# from typing import Optional, Set
#
# import pandas as pd
#
# from kbc_pul.observed_data_generation.abstract_biased_data_generation import AbstractBiasedDataGenerator
# from kbc_pul.observed_data_generation.subject_based_propensity_score_controller import \
#     SetInclusionPropensityScoreController
#
#
#
# class SARTwoSubjectGroupsBiasedDataGenerator(AbstractBiasedDataGenerator):
#
#     def __init__(self, relation_to_bias: Optional[str] = None,):
#         self.relation_to_bias: Optional[str] = relation_to_bias
#         # Get the subject entities of the filter relation
#         self.subjects_of_filter_relation_set: Set[str] = get_subject_entities_of_filter_relation(
#             df_ground_truth=df_ground_truth, filter_relation=filter_relation
#         )
#         self.verbose: bool =
#
#         self.set_inclusion_prop_score_controller = SetInclusionPropensityScoreController(
#             subject_entity_set=self.subjects_of_filter_relation_set,
#             propensity_score_if_in_set=propensity_score_subjects_of_filter_relation,
#             propensity_score_if_not_in_set=propensity_score_other_entities
#         )
#
#
#     def generate_biased_data_from_ground_truth_full(self,
#                                                     df_full_ground_truth: pd.DataFrame
#                                                     ) -> pd.DataFrame:
#         if self.relation_to_bias is None:
#             raise Exception(f"Relation to bias is unset")
#         else:
#             df_ground_truth_target_relation: pd.DataFrame = df_full_ground_truth[
#                 df_full_ground_truth["Rel"] == self.relation_to_bias
#             ]
#             df_ground_truth_non_target_relation: pd.DataFrame = df_full_ground_truth[
#                 df_full_ground_truth["Rel"] != self.relation_to_bias
#             ]
#
#
#     def generate_biased_data_from_ground_truth_single_relation(self,
#                                                                df_ground_truth_single_relation: pd.DataFrame
#                                                                ) -> pd.DataFrame:
#         if is_pca_version:
#             df_observed_target_relation: pd.DataFrame = get_df_observed_target_relation_pca(
#                 df_ground_truth_target_relation=df_ground_truth_target_relation,
#                 entity_based_propensity_score_controller=set_inclusion_prop_score_controller,
#                 rng=rng,
#                 verbose=verbose
#             )
#         else:
#             df_observed_target_relation: pd.DataFrame = get_df_observed_target_relation(
#                 df_ground_truth_target_relation=df_ground_truth_target_relation,
#                 entity_based_propensity_score_controller=set_inclusion_prop_score_controller,
#                 rng=rng
#             )
#
#         if verbose:
#             print(f"ground truth:")
#             print(f"\t{df_ground_truth.shape[0]} literals")
#             print(f"\t{df_ground_truth_target_relation.shape[0]} {target_relation} (target) literals")
#             print(f"\t{df_observed_target_relation.shape[0]} literals are selected")

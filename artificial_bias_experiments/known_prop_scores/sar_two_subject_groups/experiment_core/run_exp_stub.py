import os
import random
from typing import List, Optional, NamedTuple, Tuple
from random import Random
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from artificial_bias_experiments.experiment_utils import print_or_log


from artificial_bias_experiments.evaluation.ground_truth_utils import TrueEntitySetsTuple, \
    get_true_entity_sets_as_string_sets
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.conditioned_true_confidence_tuple import \
    ConditionedTrueConfidenceTuple, get_true_confidences_conditioned_on_filter_set, print_conditioned_true_confidences, \
    get_true_positive_pair_confidence_on_filter_set
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    KnownPropScoresSARExperimentInfo
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.observed_target_rel_two_groups_stats import \
    SARTwoGroupsObservedTargetRelStats, calculated_sar_two_groups_observed_target_rel_stats, \
    print_sar_two_groups_observed_target_rel_stats
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.trial_conditional_confidence_stats import \
    TrialConditionalTrueConfidencesManager

from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from kbc_pul.amie.amie_output_rule_extraction import get_amie_rules_from_rule_tsv_file
from kbc_pul.observed_data_generation.abstract_triple_selection import ObservedTargetRelationInfo
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_triple_selection import \
    SARSubjectGroupBasedTripleSelector, get_subject_group_based_triple_selector
from kbc_pul.observed_data_generation.sar_two_subject_groups.subject_based_propensity_score_controller import \
    SetInclusionPropensityScoreController, build_set_inclusion_propensity_score_controller_from
from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.data_structures.rule_wrapper import RuleWrapper, filter_rules_predicting
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_cwa_and_pca_confidences_from_cached_predictions import \
    set_rule_wrapper_cwa_and_pca_confidence_calculated_from_cache
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_ipw_and_ipw_pca_confidences_from_cached_predictions import  \
    calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_true_confidence_on_observed_data_from_cached_predictions import \
    calculate_true_confidence_metrics_from_df_cached_predictions

from pylo.language.lp import (Clause as PyloClause, global_context as pylo_global_context)


class KnownPropScoreSarExperimentInfo(NamedTuple):
    dataset_name: str
    target_relation: str
    filter_relation: str
    true_prop_scores: PropScoresTwoSARGroups
    is_pca_version: bool


def write_observed_target_relation_to_csv(
        mask_observed_rows: pd.Series,
        experiment_dir: str,
        experiment_info: KnownPropScoresSARExperimentInfo,
        random_trial_index: int
):
    filename_mask_observed_rows: str = os.path.join(
        experiment_dir,
        f"mask_{experiment_info.target_relation}_{experiment_info.filter_relation}"
        f"_s_prop{experiment_info.true_prop_scores.in_filter}"
        f"_ns_prop{experiment_info.true_prop_scores.other}_trial{random_trial_index}.csv"
    )
    mask_observed_rows.astype(int).to_csv(
        filename_mask_observed_rows,
        sep="\t",
        index=False,
        header=False

    )


def print_propensity_scores_and_label_frequency(
        experiment_info: KnownPropScoresSARExperimentInfo,
        observed_target_rel_stats: SARTwoGroupsObservedTargetRelStats
):
    print("Start evaluation phase")
    table = [
        [f"Prop score s for {experiment_info.filter_relation}(s, *)",
         experiment_info.true_prop_scores.in_filter],
        [f"Prop score s for NOT  {experiment_info.filter_relation}(S, *)",
         experiment_info.true_prop_scores.other],
        ["Label frequency (observed KB)", f"{observed_target_rel_stats.observed_label_frequency: 0.3f}"]
    ]
    print(tabulate(table))


def check_if_setting_has_interesting_rules(
        target_relation: str,
        amie_rule_tsv_filename: str,
        set_inclusion_prop_score_controller: SetInclusionPropensityScoreController,
        df_ground_truth_target_relation: pd.DataFrame,
        pandas_kb_wrapper: PandasKnowledgeBaseWrapper,
):
    are_interesting_rules: bool = False

    rule_wrapper_list: List[RuleWrapper] = [
        rule_wrapper.clone_with_metrics_unset()
        for rule_wrapper in filter_rules_predicting(
            get_amie_rules_from_rule_tsv_file(amie_rule_tsv_filename),
            head_functor_set={
                target_relation
            }
        )
    ]
    for rule_wrapper in rule_wrapper_list:
        o_df_predictions_on_ground_truth: Optional[pd.DataFrame] = pandas_kb_wrapper.get_predictions_for_rule(
            rule=rule_wrapper.rule)
        if o_df_predictions_on_ground_truth is None:
            pass

        o_conditioned_true_conf_tuple: Optional[ConditionedTrueConfidenceTuple]
        o_conditioned_true_conf_tuple = get_true_confidences_conditioned_on_filter_set(
            o_df_cached_predictions=o_df_predictions_on_ground_truth,
            set_incl_prop_score_controller=set_inclusion_prop_score_controller,
            df_ground_truth_target_rel=df_ground_truth_target_relation,
        )
        if o_conditioned_true_conf_tuple is None:
            return False
        else:
            if o_conditioned_true_conf_tuple.is_interesting():
                return True

    return are_interesting_rules


def do_single_random_trial(
        experiment_info: KnownPropScoresSARExperimentInfo,
        experiment_dir: str,

        pandas_kb_wrapper: PandasKnowledgeBaseWrapper,

        df_ground_truth_target_relation: pd.DataFrame,
        true_ent_sets_tuple: TrueEntitySetsTuple,
        true_set_inclusion_prop_score_controller: SetInclusionPropensityScoreController,

        amie_rule_wrappers: List[RuleWrapper],
        dir_rule_wrappers: str,

        random_trial_index: int,
        n_random_trials: int,
        rng: Random,
        verbose: bool
):
    print_or_log(message="Generating biased dataset", logger=None, verbose=verbose)
    #################################################################################################################
    # Generate observed data (SAR)
    triple_selector: SARSubjectGroupBasedTripleSelector = get_subject_group_based_triple_selector(
        use_pca_version=experiment_info.is_pca_version,
        entity_based_propensity_score_controller=true_set_inclusion_prop_score_controller,
        verbose=False
    )
    observed_target_relation_info: ObservedTargetRelationInfo = triple_selector.select_observed_target_relation(
        df_ground_truth_target_relation=df_ground_truth_target_relation,
        rng=rng
    )
    write_observed_target_relation_to_csv(
        mask_observed_rows=observed_target_relation_info.mask,
        experiment_dir=experiment_dir,
        experiment_info=experiment_info,
        random_trial_index=random_trial_index
    )

    # Set the observed target relation
    pandas_kb_wrapper.replace_predicate(
        relation=experiment_info.target_relation,
        new_df_for_relation=observed_target_relation_info.df
    )

    print_or_log(message="Finished generating biased dataset (non-PCA)", logger=None, verbose=verbose)
    print_or_log(message="Evaluating rules on biased dataset (non-PCA)", logger=None, verbose=verbose)

    # Look at how the observed target relation is divided in two groups by the filter relation
    observed_target_rel_stats: SARTwoGroupsObservedTargetRelStats
    observed_target_rel_stats = calculated_sar_two_groups_observed_target_rel_stats(
        df_ground_truth_target_relation=df_ground_truth_target_relation,
        observed_target_rel=observed_target_relation_info,
        prop_score_controller=true_set_inclusion_prop_score_controller
    )

    if verbose:
        print_sar_two_groups_observed_target_rel_stats(
            filter_relation=experiment_info.filter_relation,
            observed_target_rel_stats=observed_target_rel_stats
        )
        print_propensity_scores_and_label_frequency(
            experiment_info=experiment_info,
            observed_target_rel_stats=observed_target_rel_stats
        )

    #############################################################################################
    print(f"Start evaluation (random trial index {random_trial_index}"
          f"  ({random_trial_index + 1} / {n_random_trials}))")
    trial_cond_true_confidences_manager = TrialConditionalTrueConfidencesManager(
        experiment_info=experiment_info,
        random_trial_index=random_trial_index
    )

    rule_wrapper_list: List[RuleWrapper] = [
        rule_wrapper.clone_with_metrics_unset()
        for rule_wrapper in amie_rule_wrappers
    ]
    rule_wrapper: RuleWrapper
    for rule_wrapper in tqdm(rule_wrapper_list, disable=not verbose):
        rule: PyloClause = rule_wrapper.rule
        if verbose:
            print(f"Rule: {rule}")

        o_df_cached_predictions: Optional[pd.DataFrame] = pandas_kb_wrapper.calculate_prediction_cache_for_rule(
            rule=rule_wrapper.rule
        )

        if o_df_cached_predictions is None or len(o_df_cached_predictions) == 0:
            if verbose:
                print(f"ZERO PREDICTIONS for rule {rule}")
        else:
            # STD, PCA
            set_rule_wrapper_cwa_and_pca_confidence_calculated_from_cache(rule_wrapper, o_df_cached_predictions)
            rule_wrapper.set_inverse_c_weighted_std_confidence(
                label_frequency=observed_target_rel_stats.observed_label_frequency
            )

            # Calc True conf, true conf*
            calculate_true_confidence_metrics_from_df_cached_predictions(
                rule_wrapper=rule_wrapper,
                df_cached_predictions=o_df_cached_predictions,
                df_ground_truth_target_relation=df_ground_truth_target_relation,
                true_entity_sets=true_ent_sets_tuple
            )

            # Note: group-local TRUE (default) confidences
            o_conditioned_true_conf_tuple: Optional[ConditionedTrueConfidenceTuple]
            o_conditioned_true_conf_tuple = get_true_confidences_conditioned_on_filter_set(
                o_df_cached_predictions=o_df_cached_predictions,
                set_incl_prop_score_controller=true_set_inclusion_prop_score_controller,
                df_ground_truth_target_rel=df_ground_truth_target_relation,
            )
            if o_conditioned_true_conf_tuple is None:
                print("No predictions on one of the two rule sets")
            else:
                if verbose:
                    print_conditioned_true_confidences(
                        filter_relation=experiment_info.filter_relation,
                        conditioned_true_confidence_tuple=o_conditioned_true_conf_tuple,
                        true_confidence_on_all_predictions=rule_wrapper.o_true_confidence,
                    )

            # Note: group-local POSITIVE-PAIR confidence
            o_conditioned_true_pos_pair_tuples: Optional[
                Tuple[ConditionedTrueConfidenceTuple, ConditionedTrueConfidenceTuple]
            ] = get_true_positive_pair_confidence_on_filter_set(
                o_df_cached_predictions=o_df_cached_predictions,
                set_incl_prop_score_controller=true_set_inclusion_prop_score_controller,
                true_entity_sets=true_ent_sets_tuple
            )
            if o_conditioned_true_pos_pair_tuples is None:
                print("No predictions on one of the two rule sets (triggered with true pos-pair conf)")

            calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions(
                rule_wrapper=rule_wrapper,
                df_cached_predictions=o_df_cached_predictions,
                pylo_context=pylo_global_context,
                propensity_score_controller=true_set_inclusion_prop_score_controller,
                verbose=verbose
            )

            filename_rule_wrapper: str = os.path.join(
                dir_rule_wrappers,
                f"{str(rule_wrapper.rule)}_trial{random_trial_index}.json.gz"
            )
            rule_wrapper.to_json_file(filename_rule_wrapper)

            trial_cond_true_confidences_manager.add_rule_wrapper_info(  # AttributeError: 'NoneType' object has no attribute 'n_predictions_in_filter'
                rule_wrapper=rule_wrapper,
                conditioned_true_conf_tuple=o_conditioned_true_conf_tuple,
                conditioned_true_pos_pair_tuples=o_conditioned_true_pos_pair_tuples
            )

    trial_cond_true_confidences_manager.make_file_indicating_at_least_one_rule_was_interesting(
        experiment_dir=experiment_dir
    )
    trial_cond_true_confidences_manager.to_csv(
        experiment_dir=experiment_dir
    )

    print_or_log(message="Finished evaluating rules on biased dataset (non-PCA)", logger=None, verbose=verbose)


def run_single_experiment_setting_of_experiment_known_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        experiment_info: KnownPropScoresSARExperimentInfo,
        random_seed: int,
        n_random_trials: int,
        verbose: bool = False,
):
    df_ground_truth: pd.DataFrame = get_df_ground_truth(
        filename_ground_truth_dataset,
        separator_ground_truth_dataset
    )
    pandas_kb_wrapper = PandasKnowledgeBaseWrapper.create_from_full_data(df_full_data=df_ground_truth)
    df_ground_truth_target_relation: pd.DataFrame = pandas_kb_wrapper.get_relation(
        experiment_info.target_relation)

    if verbose:
        print(f"ground truth:")
        print(f"\t{df_ground_truth.shape[0]} literals")
        print(f"\t{df_ground_truth_target_relation.shape[0]} {experiment_info.target_relation} (target) literals")

    # PropensityScoreController
    set_inclusion_prop_score_controller: SetInclusionPropensityScoreController
    set_inclusion_prop_score_controller = build_set_inclusion_propensity_score_controller_from(
        df_ground_truth=df_ground_truth,
        filter_relation=experiment_info.filter_relation,
        prop_scores_two_groups=experiment_info.true_prop_scores,
        verbose=False
    )

    #############################################################################################
    do_interesting_rules_exist: bool = check_if_setting_has_interesting_rules(
        target_relation=experiment_info.target_relation,
        amie_rule_tsv_filename=amie_rule_tsv_filename,
        set_inclusion_prop_score_controller=set_inclusion_prop_score_controller,
        df_ground_truth_target_relation=df_ground_truth_target_relation,
        pandas_kb_wrapper=pandas_kb_wrapper
    )
    print(f"Do interesting rules exist? {do_interesting_rules_exist}  "
          f"(target: {experiment_info.target_relation}, filter: {experiment_info.filter_relation}, e(not filter): {experiment_info.true_prop_scores.other})")
    if do_interesting_rules_exist:
        experiment_dir: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_experiment_specific(
            experiment_info=experiment_info
        )

        rng = random.Random(random_seed)

        dir_rule_wrappers: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_rule_wrappers(
            experiment_info=experiment_info
        )
        if not os.path.exists(dir_rule_wrappers):
            os.makedirs(dir_rule_wrappers)

        amie_rule_wrappers: List[RuleWrapper] = [
            rule_wrapper.clone_with_metrics_unset()
            for rule_wrapper in filter_rules_predicting(
                get_amie_rules_from_rule_tsv_file(amie_rule_tsv_filename),
                head_functor_set={
                    experiment_info.target_relation
                }
            )
        ]

        true_ent_sets_tuple: TrueEntitySetsTuple = get_true_entity_sets_as_string_sets(
            df_ground_truth_target_relation
        )

        #############################################################################################
        random_trial_index: int
        for random_trial_index in range(n_random_trials):

            do_single_random_trial(
                experiment_info=experiment_info,
                experiment_dir=experiment_dir,

                pandas_kb_wrapper=pandas_kb_wrapper,

                df_ground_truth_target_relation=df_ground_truth_target_relation,
                true_ent_sets_tuple=true_ent_sets_tuple,
                true_set_inclusion_prop_score_controller=set_inclusion_prop_score_controller,

                amie_rule_wrappers=amie_rule_wrappers,
                dir_rule_wrappers=dir_rule_wrappers,

                random_trial_index=random_trial_index,
                n_random_trials=n_random_trials,
                rng=rng,
                verbose=verbose
            )

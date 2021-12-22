import os
import random
from typing import Optional, List, Tuple
from random import Random
import pandas as pd
from pylo.language.lp import (Clause as PyloClause, global_context as pylo_global_context)
from tqdm import tqdm
from artificial_bias_experiments.experiment_utils import print_or_log

from kbc_pul.amie.amie_output_rule_extraction import get_amie_rules_from_rule_tsv_file
from artificial_bias_experiments.evaluation.ground_truth_utils import TrueEntitySetsTuple, \
    get_true_entity_sets_as_string_sets
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.conditioned_true_confidence_tuple import \
    ConditionedTrueConfidenceTuple, get_true_confidences_conditioned_on_filter_set, print_conditioned_true_confidences, \
    get_true_positive_pair_confidence_on_filter_set

from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_core.run_exp_stub import \
    check_if_setting_has_interesting_rules, print_propensity_scores_and_label_frequency
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.observed_target_rel_two_groups_stats import \
    SARTwoGroupsObservedTargetRelStats, calculated_sar_two_groups_observed_target_rel_stats, \
    print_sar_two_groups_observed_target_rel_stats
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.trial_conditional_confidence_stats import \
    TrialConditionalTrueConfidencesManager
from artificial_bias_experiments.noisy_prop_scores.available_prop_scores_to_pu_metrics_controller import \
    RuleWrapperNoisyPropScoresToPuMetricsMap
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_info import \
    NoisyPropScoresSARExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.noisy_prop_scores_sar_two_groups_file_naming import \
    NoisyPropScoresSARTwoGroupsFileNamer
from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.data_structures.rule_wrapper import filter_rules_predicting, RuleWrapper
from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from kbc_pul.observed_data_generation.abstract_triple_selection import ObservedTargetRelationInfo
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_triple_selection import \
    SARSubjectGroupBasedTripleSelector, get_subject_group_based_triple_selector
from kbc_pul.observed_data_generation.sar_two_subject_groups.subject_based_propensity_score_controller import \
    SetInclusionPropensityScoreController, build_set_inclusion_propensity_score_controller_from
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_cwa_and_pca_confidences_from_cached_predictions import \
    set_rule_wrapper_cwa_and_pca_confidence_calculated_from_cache
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_ipw_and_ipw_pca_confidences_from_cached_predictions import \
    calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_true_confidence_on_observed_data_from_cached_predictions import \
    calculate_true_confidence_metrics_from_df_cached_predictions

PropScore = float


def write_observed_target_relation_to_csv(
        mask_observed_rows: pd.Series,
        experiment_dir: str,
        experiment_info: NoisyPropScoresSARExperimentInfo,
        random_trial_index: int
) -> None:
    filename_mask_observed_rows: str = os.path.join(
        experiment_dir,
        f"mask_{experiment_info.target_relation}"
        f"s_prop{experiment_info.true_prop_scores.in_filter}"
        f"_ns_prop{experiment_info.true_prop_scores.other}"
        f"_trial{random_trial_index}.csv"
    )
    mask_observed_rows.astype(int).to_csv(
        filename_mask_observed_rows,
        sep="\t",
        index=False,
        header=False

    )


def do_single_random_trial(
        experiment_info: NoisyPropScoresSARExperimentInfo,
        experiment_dir: str,

        pandas_kb_wrapper: PandasKnowledgeBaseWrapper,

        df_ground_truth_target_relation: pd.DataFrame,
        true_ent_sets_tuple: TrueEntitySetsTuple,
        true_set_inclusion_prop_score_controller: SetInclusionPropensityScoreController,

        amie_rule_wrappers: List[RuleWrapper],
        dir_rule_wrappers: str,
        dir_pu_metrics_of_rule_wrappers: str,

        random_trial_index: int,
        n_random_trials: int,
        rng: Random,
        verbose: bool
):
    print_or_log(message="Generating SAR dataset", logger=None, verbose=verbose)
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

    print_or_log(message="Finished generating SAR dataset", logger=None, verbose=verbose)
    print_or_log(message="Evaluating rules on SAR dataset", logger=None, verbose=verbose)

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
    print(f"Checking out {len(rule_wrapper_list)} rules")
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

            # TRUE CONF & CONF*
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
                print(f"No predictions on one of the two rule sets for rule {rule_wrapper.rule}")
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

                filename_rule_wrapper: str = os.path.join(
                    dir_rule_wrappers,
                    f"{str(rule_wrapper.rule)}_trial{random_trial_index}.json.gz"
                )
                rule_wrapper.to_json_file(filename_rule_wrapper)



                trial_cond_true_confidences_manager.add_rule_wrapper_info(
                    # AttributeError: 'NoneType' object has no attribute 'n_predictions_in_filter'
                    rule_wrapper=rule_wrapper,
                    conditioned_true_conf_tuple=o_conditioned_true_conf_tuple,
                    conditioned_true_pos_pair_tuples=o_conditioned_true_pos_pair_tuples
                )

                ##################################################
                noisy_prop_score_tuple_to_pu_metrics_map_controller = RuleWrapperNoisyPropScoresToPuMetricsMap(
                    rule_str=str(rule_wrapper.rule),
                    random_trial_index=random_trial_index,
                    true_prop_scores=experiment_info.true_prop_scores
                )
                noisy_propensity_score_other_entities: float
                for noisy_propensity_score_other_entities in experiment_info.noisy_prop_score_not_in_filter_list:
                    rule_wrapper.o_relative_pu_confidence_unbiased = None
                    rule_wrapper.o_relative_pu_confidence_pca_subject_to_object = None
                    rule_wrapper.o_relative_pu_confidence_pca_object_to_subject = None
                    rule_wrapper.o_c_weighted_std_conf = None

                    noisy_set_inclusion_prop_score_controller = SetInclusionPropensityScoreController(
                        subject_entity_set=true_set_inclusion_prop_score_controller.get_subject_entity_set(),
                        propensity_score_if_in_set=experiment_info.noisy_prop_score_in_filter,
                        propensity_score_if_not_in_set=noisy_propensity_score_other_entities
                    )
                    calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions(
                        rule_wrapper=rule_wrapper,
                        df_cached_predictions=o_df_cached_predictions,
                        pylo_context=pylo_global_context,
                        propensity_score_controller=noisy_set_inclusion_prop_score_controller,
                        verbose=verbose
                    )

                    estimated_label_frequency: float = noisy_set_inclusion_prop_score_controller.get_label_frequency_given_mask_observed_literals_in_set(
                        mask_observed_literals_in_set=observed_target_rel_stats.mask_observed_literals_in_filter_set
                    )
                    rule_wrapper.set_inverse_c_weighted_std_confidence(
                        label_frequency=observed_target_rel_stats.observed_label_frequency
                    )
                    rule_wrapper.set_inverse_c_weighted_std_confidence(
                        label_frequency=estimated_label_frequency
                    )

                    noisy_prop_score_tuple_to_pu_metrics_map_controller.add_pu_metrics_for_available_prop_scores(
                        noisy_prop_scores=PropScoresTwoSARGroups(
                            in_filter=experiment_info.noisy_prop_score_in_filter,
                            other=noisy_propensity_score_other_entities
                        ),
                        o_relative_pu_confidence_unbiased=rule_wrapper.o_relative_pu_confidence_unbiased,
                        o_relative_pu_confidence_pca_subject_to_object=rule_wrapper.o_relative_pu_confidence_pca_subject_to_object,
                        o_relative_pu_confidence_pca_object_to_subject=rule_wrapper.o_relative_pu_confidence_pca_object_to_subject,
                        o_inverse_c_weighted_std_confidence=rule_wrapper.o_c_weighted_std_conf
                    )

                filename_pu_metrics_of_rule_wrapper: str = os.path.join(
                    dir_pu_metrics_of_rule_wrappers,
                    f"{str(rule_wrapper.rule)}_trial{random_trial_index}.tsv.gz"
                )
                noisy_prop_score_tuple_to_pu_metrics_map_controller.to_tsv(
                    filename_noisy_prop_scores_to_pu_metrics_map=filename_pu_metrics_of_rule_wrapper
                )
    trial_cond_true_confidences_manager.make_file_indicating_at_least_one_rule_was_interesting(
        experiment_dir=experiment_dir
    )
    trial_cond_true_confidences_manager.to_csv(
        experiment_dir=experiment_dir
    )


def run_single_experiment_setting_of_experiment_noisy_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        experiment_info: NoisyPropScoresSARExperimentInfo,
        random_seed: int,
        n_random_trials: int,
        verbose: bool = False
) -> None:
    df_ground_truth: pd.DataFrame = get_df_ground_truth(
        filename_ground_truth_dataset, separator_ground_truth_dataset
    )

    pandas_kb_wrapper = PandasKnowledgeBaseWrapper.create_from_full_data(df_full_data=df_ground_truth)
    df_ground_truth_target_relation: pd.DataFrame = pandas_kb_wrapper.get_relation(
        experiment_info.target_relation
    )

    if verbose:
        print(f"ground truth:")
        print(f"\t{df_ground_truth.shape[0]} literals")
        print(f"\t{df_ground_truth_target_relation.shape[0]} {experiment_info.target_relation} (target) literals")

    # PropensityScoreController
    true_set_inclusion_prop_score_controller: SetInclusionPropensityScoreController
    true_set_inclusion_prop_score_controller = build_set_inclusion_propensity_score_controller_from(
        df_ground_truth=df_ground_truth,
        filter_relation=experiment_info.filter_relation,
        prop_scores_two_groups=experiment_info.true_prop_scores,
        verbose=False
    )

    do_interesting_rules_exist: bool = check_if_setting_has_interesting_rules(
        target_relation=experiment_info.target_relation,
        amie_rule_tsv_filename=amie_rule_tsv_filename,
        set_inclusion_prop_score_controller=true_set_inclusion_prop_score_controller,
        df_ground_truth_target_relation=df_ground_truth_target_relation,
        pandas_kb_wrapper=pandas_kb_wrapper
    )
    print(f"Do interesting rules exist "
          f"({experiment_info.target_relation} - {experiment_info.filter_relation})?"
          f" {do_interesting_rules_exist}")
    if do_interesting_rules_exist:
        experiment_dir: str = NoisyPropScoresSARTwoGroupsFileNamer.get_dir_experiment_specific(
            experiment_info=experiment_info
        )

        rng = random.Random(random_seed)

        dir_rule_wrappers: str = os.path.join(
            experiment_dir,
            'rule_wrappers'
        )
        if not os.path.exists(dir_rule_wrappers):
            os.makedirs(dir_rule_wrappers)

        dir_pu_metrics_of_rule_wrappers: str = os.path.join(
            dir_rule_wrappers,
            "pu_metrics"
        )
        if not os.path.exists(dir_pu_metrics_of_rule_wrappers):
            os.makedirs(dir_pu_metrics_of_rule_wrappers)

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
                true_set_inclusion_prop_score_controller=true_set_inclusion_prop_score_controller,

                amie_rule_wrappers=amie_rule_wrappers,
                dir_rule_wrappers=dir_rule_wrappers,
                dir_pu_metrics_of_rule_wrappers=dir_pu_metrics_of_rule_wrappers,

                random_trial_index=random_trial_index,
                n_random_trials=n_random_trials,
                rng=rng,
                verbose=verbose

            )

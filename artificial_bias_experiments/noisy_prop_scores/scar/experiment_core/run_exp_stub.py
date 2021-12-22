import os
import random
from typing import List, Optional

import pandas as pd
from pylo.language.lp import (Clause as PyloClause, global_context as pylo_global_context)
from tqdm import tqdm
from artificial_bias_experiments.experiment_utils import print_or_log

from kbc_pul.amie.amie_output_rule_extraction import get_amie_rules_from_rule_tsv_file

from artificial_bias_experiments.evaluation.ground_truth_utils import TrueEntitySetsTuple, \
    get_true_entity_sets_as_string_sets
from artificial_bias_experiments.noisy_prop_scores.available_prop_scores_to_pu_metrics_controller import \
    RuleWrapperNoisyPropScoresToPuMetricsMap
from artificial_bias_experiments.noisy_prop_scores.scar.experiment_info import \
    NoisyPropScoresSCARExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.scar.noisy_prop_scores_scar_file_naming import \
    NoisyPropScoresSCARFileNamer
from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.data_structures.rule_wrapper import RuleWrapper, filter_rules_predicting
from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from kbc_pul.observed_data_generation.abstract_triple_selection import ObservedTargetRelationInfo
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups
from kbc_pul.observed_data_generation.scar.scar_propensity_score_controller import \
    SCARPropensityScoreController
from kbc_pul.observed_data_generation.scar.scar_triple_selection import SCARTripleSelector
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_cwa_and_pca_confidences_from_cached_predictions import \
    set_rule_wrapper_cwa_and_pca_confidence_calculated_from_cache
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_ipw_and_ipw_pca_confidences_from_cached_predictions import \
    calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_true_confidence_on_observed_data_from_cached_predictions import \
    calculate_true_confidence_metrics_from_df_cached_predictions


def write_observed_target_relation_to_csv(
        mask_observed_rows: pd.Series,
        experiment_dir: str,
        experiment_info: NoisyPropScoresSCARExperimentInfo,
        random_trial_index: int
):
    filename_mask_observed_rows: str = os.path.join(
        experiment_dir,
        f"mask_{experiment_info.target_relation}"
        f"c{experiment_info.true_label_frequency}_trial{random_trial_index}.csv"
    )
    mask_observed_rows.astype(int).to_csv(
        filename_mask_observed_rows,
        sep="\t",
        index=False,
        header=False

    )


def run_single_experiment_setting_of_experiment_noisy_prop_scores_scar(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        experiment_info: NoisyPropScoresSCARExperimentInfo,
        random_seed: int,
        n_random_trials: int,
        verbose: bool = False,
):
    experiment_dir: str = NoisyPropScoresSCARFileNamer.get_dir_experiment_specific(
        experiment_info=experiment_info
    )
    print(experiment_dir)

    if verbose:
        print(experiment_dir)

    rng = random.Random(random_seed)
    df_ground_truth: pd.DataFrame = get_df_ground_truth(
        filename_ground_truth_dataset, separator_ground_truth_dataset
    )

    print_or_log(message="Generating SCAR dataset", logger=None, verbose=verbose)

    pandas_kb_wrapper = PandasKnowledgeBaseWrapper.create_from_full_data(df_full_data=df_ground_truth)
    df_ground_truth_target_relation: pd.DataFrame = pandas_kb_wrapper.get_relation(
        experiment_info.target_relation
    )

    if verbose:
        print(f"ground truth:")
        print(f"\t{df_ground_truth.shape[0]} literals")
        print(f"\t{df_ground_truth_target_relation.shape[0]} {experiment_info.target_relation} (target) literals")

    dir_rule_wrappers: str = os.path.join(
        experiment_dir,
        'rule_wrappers'
    )
    if not os.path.exists(dir_rule_wrappers):
        os.makedirs(dir_rule_wrappers)

    triple_selector: SCARTripleSelector = SCARTripleSelector(
        constant_label_frequency=experiment_info.true_label_frequency,
        verbose=False
    )
    #############################################################################################
    random_trial_index: int
    for random_trial_index in range(n_random_trials):

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

        true_ent_sets_tuple: TrueEntitySetsTuple = get_true_entity_sets_as_string_sets(
            df_ground_truth_target_relation
        )
        #############################################################################################

        if verbose:
            print("Start evaluation phase")
            print(f"True Label frequency: {experiment_info.true_label_frequency}")

        rule_wrapper_list: List[RuleWrapper] = [
            rule_wrapper.clone_with_metrics_unset()
            for rule_wrapper in filter_rules_predicting(
                get_amie_rules_from_rule_tsv_file(amie_rule_tsv_filename),
                head_functor_set={
                    experiment_info.target_relation
                }
            )
        ]

        print(f"Start evaluation (random trial index {random_trial_index}"
              f"  ({random_trial_index + 1} / {n_random_trials}))")
        rule_wrapper: RuleWrapper
        for rule_wrapper in tqdm(rule_wrapper_list, disable=not verbose):
            rule: PyloClause = rule_wrapper.rule
            if verbose:
                print(f"Rule: {rule}")

            o_df_cached_predictions: Optional[pd.DataFrame] = pandas_kb_wrapper.calculate_prediction_cache_for_rule(
                rule=rule_wrapper.rule
            )
            if o_df_cached_predictions is None or (len(o_df_cached_predictions) == 0):
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

                filename_rule_wrapper: str = os.path.join(
                    dir_rule_wrappers,
                    f"{str(rule_wrapper.rule)}_trial{random_trial_index}.json.gz"
                )
                rule_wrapper.to_json_file(filename_rule_wrapper)

                ##################################################

                dir_pu_metrics_of_rule_wrappers: str = os.path.join(
                    dir_rule_wrappers,
                    "pu_metrics"
                )
                if not os.path.exists(dir_pu_metrics_of_rule_wrappers):
                    os.makedirs(dir_pu_metrics_of_rule_wrappers)
                filename_pu_metrics_of_rule_wrapper: str = os.path.join(
                    dir_pu_metrics_of_rule_wrappers,
                    f"{str(rule_wrapper.rule)}_trial{random_trial_index}.tsv.gz"
                )

                noisy_prop_score_tuple_to_pu_metrics_map_controller = RuleWrapperNoisyPropScoresToPuMetricsMap(
                    rule_str=str(rule_wrapper.rule),
                    random_trial_index=random_trial_index,
                    true_prop_scores=PropScoresTwoSARGroups(
                        in_filter=experiment_info.true_label_frequency,
                        other=experiment_info.true_label_frequency
                    )
                )
                noisy_propensity_score_other_entities: float
                for noisy_label_frequency in experiment_info.available_label_frequency_list:
                    rule_wrapper.o_relative_pu_confidence_unbiased = None
                    rule_wrapper.o_relative_pu_confidence_pca_subject_to_object = None
                    rule_wrapper.o_relative_pu_confidence_pca_object_to_subject = None
                    rule_wrapper.o_c_weighted_std_conf = None

                    available_label_frequency_scar_propensity_score_controller = SCARPropensityScoreController(
                        constant_label_frequency=noisy_label_frequency
                    )
                    calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions(
                        rule_wrapper=rule_wrapper,
                        df_cached_predictions=o_df_cached_predictions,
                        pylo_context=pylo_global_context,
                        propensity_score_controller=available_label_frequency_scar_propensity_score_controller,
                        verbose=verbose
                    )

                    rule_wrapper.set_inverse_c_weighted_std_confidence(
                        label_frequency=noisy_label_frequency
                    )

                    noisy_prop_score_tuple_to_pu_metrics_map_controller.add_pu_metrics_for_available_prop_scores(
                        noisy_prop_scores=PropScoresTwoSARGroups(
                            in_filter=noisy_label_frequency,
                            other=noisy_label_frequency
                        ),
                        o_relative_pu_confidence_unbiased=rule_wrapper.o_relative_pu_confidence_unbiased,
                        o_relative_pu_confidence_pca_subject_to_object=rule_wrapper.o_relative_pu_confidence_pca_subject_to_object,
                        o_relative_pu_confidence_pca_object_to_subject=rule_wrapper.o_relative_pu_confidence_pca_object_to_subject,
                        o_inverse_c_weighted_std_confidence=rule_wrapper.o_c_weighted_std_conf
                    )
                noisy_prop_score_tuple_to_pu_metrics_map_controller.to_tsv(
                    filename_noisy_prop_scores_to_pu_metrics_map=filename_pu_metrics_of_rule_wrapper
                )

        print_or_log(message="Finished evaluating rules on biased dataset (non-PCA)", logger=None, verbose=verbose)

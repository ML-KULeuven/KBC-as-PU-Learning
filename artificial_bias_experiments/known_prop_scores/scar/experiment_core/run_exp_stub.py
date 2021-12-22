import os
import random
from typing import List, Optional

import pandas as pd
from pylo.language.lp import (Clause as PyloClause, global_context as pylo_global_context)
from tqdm import tqdm
from artificial_bias_experiments.experiment_utils import print_or_log
from artificial_bias_experiments.known_prop_scores.scar.known_prop_scores_scar_file_naming import \
    KnownPropScoresSCARConstantLabelFreqFileNamer
from artificial_bias_experiments.known_prop_scores.scar.experiment_info import \
    KnownPropScoresSCARExperimentInfo
from artificial_bias_experiments.evaluation.ground_truth_utils import TrueEntitySetsTuple, \
    get_true_entity_sets_as_string_sets

from kbc_pul.amie.amie_output_rule_extraction import get_amie_rules_from_rule_tsv_file
from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.data_structures.rule_wrapper import RuleWrapper, filter_rules_predicting
from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from kbc_pul.observed_data_generation.abstract_triple_selection import ObservedTargetRelationInfo
from kbc_pul.observed_data_generation.scar.scar_propensity_score_controller import \
    SCARPropensityScoreController
from kbc_pul.observed_data_generation.scar.scar_triple_selection import SCARTripleSelector
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_cwa_and_pca_confidences_from_cached_predictions import \
    set_rule_wrapper_cwa_and_pca_confidence_calculated_from_cache
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_ipw_and_ipw_pca_confidences_from_cached_predictions import  \
    calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_true_confidence_on_observed_data_from_cached_predictions import \
    calculate_true_confidence_metrics_from_df_cached_predictions


def write_observed_target_relation_to_csv(
        mask_observed_rows: pd.Series,
        experiment_dir: str,
        experiment_info: KnownPropScoresSCARExperimentInfo,
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


def run_single_experiment_setting_of_experiment_known_prop_scores_scar(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        experiment_info: KnownPropScoresSCARExperimentInfo,
        random_seed: int,
        n_random_trials: int,
        verbose: bool = False,
):
    experiment_dir: str = KnownPropScoresSCARConstantLabelFreqFileNamer.get_dir_experiment_specific(
        experiment_info=experiment_info
    )
    print(experiment_dir)

    rng = random.Random(random_seed)
    df_ground_truth: pd.DataFrame = get_df_ground_truth(filename_ground_truth_dataset, separator_ground_truth_dataset)

    print_or_log(message="Generating SCAR dataset (non-PCA)", logger=None, verbose=verbose)

    pandas_kb_wrapper = PandasKnowledgeBaseWrapper.create_from_full_data(df_full_data=df_ground_truth)
    df_ground_truth_target_relation: pd.DataFrame = pandas_kb_wrapper.get_relation(
        experiment_info.target_relation
    )

    if verbose:
        print(f"ground truth:")
        print(f"\t{df_ground_truth.shape[0]} literals")
        print(f"\t{df_ground_truth_target_relation.shape[0]} {experiment_info.target_relation} (target) literals")

    dir_rule_wrappers: str = KnownPropScoresSCARConstantLabelFreqFileNamer.get_dir_rule_wrappers(
        experiment_info=experiment_info
    )
    if not os.path.exists(dir_rule_wrappers):
        os.makedirs(dir_rule_wrappers)

    scar_propensity_score_controller = SCARPropensityScoreController(
        constant_label_frequency=experiment_info.true_label_frequency
    )

    #############################################################################################
    random_trial_index: int
    for random_trial_index in range(n_random_trials):
        triple_selector: SCARTripleSelector = SCARTripleSelector(
            constant_label_frequency=experiment_info.true_label_frequency,
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
            print(f"Label frequency: {experiment_info.true_label_frequency}")

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
            if o_df_cached_predictions is None or len(o_df_cached_predictions) == 0:
                if verbose:
                    print(f"ZERO PREDICTIONS for rule {rule}")
            else:
                set_rule_wrapper_cwa_and_pca_confidence_calculated_from_cache(rule_wrapper, o_df_cached_predictions)

                calculate_true_confidence_metrics_from_df_cached_predictions(
                    rule_wrapper=rule_wrapper,
                    df_cached_predictions=o_df_cached_predictions,
                    df_ground_truth_target_relation=df_ground_truth_target_relation,
                    true_entity_sets=true_ent_sets_tuple
                )

                rule_wrapper.set_inverse_c_weighted_std_confidence(
                    experiment_info.true_label_frequency
                )

                calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions(
                    rule_wrapper=rule_wrapper,
                    df_cached_predictions=o_df_cached_predictions,
                    pylo_context=pylo_global_context,
                    propensity_score_controller=scar_propensity_score_controller,
                    verbose=verbose
                )

                filename_rule_wrapper: str = os.path.join(
                    dir_rule_wrappers,
                    f"{str(rule_wrapper.rule)}_trial{random_trial_index}.json.gz"
                )
                rule_wrapper.to_json_file(filename_rule_wrapper)

        print_or_log(message="Finished evaluating rules on biased dataset (non-PCA)", logger=None, verbose=verbose)

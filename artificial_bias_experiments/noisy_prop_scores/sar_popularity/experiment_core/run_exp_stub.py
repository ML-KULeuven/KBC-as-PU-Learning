import os
import random
from typing import List, Optional
from random import Random
import pandas as pd
from tqdm import tqdm
from artificial_bias_experiments.experiment_utils import print_or_log

from kbc_pul.amie.amie_output_rule_extraction import get_amie_rules_from_rule_tsv_file
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.experiment_info import \
    NoisyPropScoresSARPopularityExperimentInfo

from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noise_definition import \
    NoiseDefinition
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noise_level_to_pu_metrics_controller import \
    RuleWrapperNoiseLevelToPuMetricsMap
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noisy_prop_scores_sar_popularity_file_naming import \
    NoisyPropScoresSARPopularityFileNamer

from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from artificial_bias_experiments.evaluation.ground_truth_utils import TrueEntitySetsTuple, \
    get_true_entity_sets_as_string_sets
from kbc_pul.observed_data_generation.abstract_triple_selection import ObservedTargetRelationInfo

from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.data_structures.rule_wrapper import RuleWrapper, filter_rules_predicting
from kbc_pul.popularity.entity_count_db import EntityCountDB
from kbc_pul.popularity.entity_counting.count_to_normalized_popularity import \
    LogisticCountToNormalizedPopularityMapper
from kbc_pul.popularity.entity_counting.ent_count_in_other_rels_both_positions import \
    EntityCountInOtherRelationsBothPositions
from kbc_pul.popularity.entity_counting.total_ent_count import EntityTripleCountController
from kbc_pul.popularity.sar_popularity_based_triple_selection import PopularityBasedTripleSelectorNonPCA
from kbc_pul.popularity.subject_popularity_based_propensity_score_controller import \
    SubjectPopularityBasedPropensityScoreController
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_cwa_and_pca_confidences_from_cached_predictions import \
    set_rule_wrapper_cwa_and_pca_confidence_calculated_from_cache
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_ipw_and_ipw_pca_confidences_from_cached_predictions import  \
    calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions
from kbc_pul.rule_metrics.prediction_cache_rule_metrics.rule_true_confidence_on_observed_data_from_cached_predictions import \
    calculate_true_confidence_metrics_from_df_cached_predictions

from pylo.language.lp import (Clause as PyloClause, global_context as pylo_global_context)


def write_observed_target_relation_to_csv(
        mask_observed_rows: pd.Series,
        experiment_dir: str,
        experiment_info: NoisyPropScoresSARPopularityExperimentInfo,
        random_trial_index: int
):
    filename_mask_observed_rows: str = os.path.join(
        experiment_dir,
        f"mask_{experiment_info.target_relation}"
        f"_log_growth{experiment_info.log_growth_rate}"
        f"_trial{random_trial_index}.csv"
    )
    mask_observed_rows.astype(int).to_csv(
        filename_mask_observed_rows,
        sep="\t",
        index=False,
        header=False

    )


def do_single_random_trial(
        experiment_info: NoisyPropScoresSARPopularityExperimentInfo,
        experiment_dir: str,

        pandas_kb_wrapper: PandasKnowledgeBaseWrapper,

        df_ground_truth_target_relation: pd.DataFrame,
        true_ent_sets_tuple: TrueEntitySetsTuple,
        true_popularity_based_propensity_score_controller: SubjectPopularityBasedPropensityScoreController,
        # additive_noise_levels: List[float],

        amie_rule_wrappers: List[RuleWrapper],
        dir_rule_wrappers: str,
        dir_pu_metrics_of_rule_wrappers: str,

        random_trial_index: int,
        n_random_trials: int,
        rng: Random,
        verbose: bool
):
    print_or_log(message="Generating biased dataset", logger=None, verbose=verbose)
    #################################################################################################################
    # Generate observed data (SAR)
    triple_selector: PopularityBasedTripleSelectorNonPCA = PopularityBasedTripleSelectorNonPCA(
        popularity_based_propensity_score_controller=true_popularity_based_propensity_score_controller,
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

    noise_definition: NoiseDefinition = experiment_info.noise_type.get_noise_definition()

    #############################################################################################
    print(f"Start evaluation {experiment_info.target_relation}, k: {experiment_info.log_growth_rate} (random trial index {random_trial_index}"
          f"  ({random_trial_index + 1} / {n_random_trials}))")

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
            # rule_wrapper.set_inverse_c_weighted_std_confidence(
            #     label_frequency=triple_selector.o_label_frequency_observed_relation
            # )

            # Calc True conf, true conf*
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

            noise_level_to_pu_metrics_map_controller = RuleWrapperNoiseLevelToPuMetricsMap(
                rule_str=str(rule_wrapper.rule),
                random_trial_index=random_trial_index,
                log_growth_rate=experiment_info.log_growth_rate
            )
            noise_level: float
            for noise_level in experiment_info.noise_levels:
                rule_wrapper.o_relative_pu_confidence_unbiased = None
                rule_wrapper.o_relative_pu_confidence_pca_subject_to_object = None
                rule_wrapper.o_relative_pu_confidence_pca_object_to_subject = None
                rule_wrapper.o_c_weighted_std_conf = None

                noisy_propensity_score_controller = noise_definition.get_noisy_propensity_score_controller(
                    noise_level=noise_level,
                    true_log_growth_rate=experiment_info.log_growth_rate,
                    true_propensity_score_controller=true_popularity_based_propensity_score_controller
                )

                # noisy_propensity_score_controller = ConstantAddedNoisePropensityScoreController(
                #     constant_additive_noise=noise_level,
                #     true_propensity_score_controller=true_popularity_based_propensity_score_controller
                # )
                noisy_label_frequency = noisy_propensity_score_controller.get_label_frequency_given_df_observed_literals(
                    df_observed_target_relation=observed_target_relation_info.df
                )
                rule_wrapper.set_inverse_c_weighted_std_confidence(
                    label_frequency=noisy_label_frequency
                )
                calculate_rule_ipw_and_ipw_pca_confidences_from_df_cached_predictions(
                    rule_wrapper=rule_wrapper,
                    df_cached_predictions=o_df_cached_predictions,
                    pylo_context=pylo_global_context,
                    propensity_score_controller=noisy_propensity_score_controller,
                    verbose=verbose
                )
                noise_level_to_pu_metrics_map_controller.add_pu_metrics_for_noise_level(
                    noise_level=noise_level,
                    o_relative_pu_confidence_unbiased=rule_wrapper.o_relative_pu_confidence_unbiased,
                    o_relative_pu_confidence_pca_subject_to_object=rule_wrapper.o_relative_pu_confidence_pca_subject_to_object,
                    o_relative_pu_confidence_pca_object_to_subject=rule_wrapper.o_relative_pu_confidence_pca_object_to_subject,
                    o_inverse_c_weighted_std_confidence=rule_wrapper.o_c_weighted_std_conf
                )
            filename_pu_metrics_of_rule_wrapper: str = os.path.join(
                dir_pu_metrics_of_rule_wrappers,
                f"{str(rule_wrapper.rule)}_trial{random_trial_index}.tsv.gz"
            )
            noise_level_to_pu_metrics_map_controller.to_tsv(
                filename_noise_level_to_pu_metrics_map=filename_pu_metrics_of_rule_wrapper
            )

    print_or_log(message="Finished evaluating rules on biased dataset (non-PCA)", logger=None, verbose=verbose)


def run_single_experiment_setting_of_noisy_prop_scores_sar_popularity(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        experiment_info:  NoisyPropScoresSARPopularityExperimentInfo,
        random_seed: int,
        n_random_trials: int,
        verbose: bool = False,
        o_specific_entity_count_aggregator: Optional[EntityCountInOtherRelationsBothPositions] = None
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

    #####################################################################################################
    # PropensityScoreController
    if o_specific_entity_count_aggregator is not None:
        specific_entity_count_aggregator = o_specific_entity_count_aggregator
    else:
        ent_count_db = EntityCountDB(pandas_kb_wrapper)
        entity_triple_count_controller = EntityTripleCountController(ent_count_db)
        specific_entity_count_aggregator = EntityCountInOtherRelationsBothPositions(
            target_relation=experiment_info.target_relation,
            entity_triple_count_controller=entity_triple_count_controller
        )
        specific_entity_count_aggregator.materialize_entity_counts()

    count_to_normalized_popularity_mapper: LogisticCountToNormalizedPopularityMapper = LogisticCountToNormalizedPopularityMapper(
        log_growth_rate=experiment_info.log_growth_rate
    )

    popularity_based_propensity_score_controller = SubjectPopularityBasedPropensityScoreController(
        entity_count_aggregator=specific_entity_count_aggregator,
        count_to_normalized_popularity_mapper=count_to_normalized_popularity_mapper,
        verbose=False
    )
    ###################################################################################

    experiment_dir: str = NoisyPropScoresSARPopularityFileNamer.get_dir_experiment_specific(
        experiment_info=experiment_info
    )

    rng = random.Random(random_seed)

    dir_rule_wrappers: str = NoisyPropScoresSARPopularityFileNamer.get_dir_rule_wrappers(
        experiment_info=experiment_info
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
            true_popularity_based_propensity_score_controller=popularity_based_propensity_score_controller,
            amie_rule_wrappers=amie_rule_wrappers,
            dir_rule_wrappers=dir_rule_wrappers,
            dir_pu_metrics_of_rule_wrappers=dir_pu_metrics_of_rule_wrappers,

            random_trial_index=random_trial_index,
            n_random_trials=n_random_trials,
            rng=rng,
            verbose=verbose
        )

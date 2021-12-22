from typing import List

import pandas as pd

from artificial_bias_experiments.noisy_prop_scores.sar_popularity.experiment_core.run_exp_stub import \
    run_single_experiment_setting_of_noisy_prop_scores_sar_popularity
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.experiment_info import \
    NoisyPropScoresSARPopularityExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noise_definition import \
    NoiseTypeSARPopularity
from kbc_pul.data_structures.pandas_kb import PandasKnowledgeBaseWrapper
from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from kbc_pul.popularity.entity_count_db import EntityCountDB
from kbc_pul.popularity.entity_counting.ent_count_in_other_rels_both_positions import \
    EntityCountInOtherRelationsBothPositions
from kbc_pul.popularity.entity_counting.total_ent_count import EntityTripleCountController


def run_per_log_growth_rate_noisy_prop_scores_sar_popularity(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        dataset_name: str,
        target_relation: str,
        log_growth_rate_list: List[float],
        noise_type: NoiseTypeSARPopularity,
        noise_levels: List[float],
        random_seed: int,
        n_random_trials: int,
        is_pca_version: bool,
        verbose: bool
):
    df_ground_truth: pd.DataFrame = get_df_ground_truth(
        filename_ground_truth_dataset,
        separator_ground_truth_dataset
    )
    pandas_kb_wrapper = PandasKnowledgeBaseWrapper.create_from_full_data(df_full_data=df_ground_truth)

    #####################################################################################################
    # PropensityScoreController
    ent_count_db = EntityCountDB(pandas_kb_wrapper)
    entity_triple_count_controller = EntityTripleCountController(ent_count_db)
    specific_entity_count_aggregator = EntityCountInOtherRelationsBothPositions(
        target_relation=target_relation,
        entity_triple_count_controller=entity_triple_count_controller
    )
    specific_entity_count_aggregator.materialize_entity_counts()

    for log_growth_rate in log_growth_rate_list:
        experiment_info = NoisyPropScoresSARPopularityExperimentInfo(
            dataset_name=dataset_name,
            target_relation=target_relation,
            is_pca_version=is_pca_version,
            log_growth_rate=log_growth_rate,
            noise_type=noise_type,
            noise_levels=noise_levels,

        )

        run_single_experiment_setting_of_noisy_prop_scores_sar_popularity(
            filename_ground_truth_dataset=filename_ground_truth_dataset,
            separator_ground_truth_dataset=separator_ground_truth_dataset,
            amie_rule_tsv_filename=amie_rule_tsv_filename,
            experiment_info=experiment_info,
            random_seed=random_seed,
            n_random_trials=n_random_trials,
            verbose=verbose,
            o_specific_entity_count_aggregator=specific_entity_count_aggregator
        )

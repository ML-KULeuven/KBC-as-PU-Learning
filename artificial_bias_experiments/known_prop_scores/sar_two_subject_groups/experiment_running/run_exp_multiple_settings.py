from typing import List

from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    KnownPropScoresSARExperimentInfo
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_core.run_exp_stub import \
    run_single_experiment_setting_of_experiment_known_prop_scores_sar_two_subject_groups
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


def run_per_target_relation_experiment_known_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        dataset_name: str,
        target_relation: str,
        filter_relation_list: List[str],
        propensity_score_subjects_of_filter_relation: float,
        propensity_score_other_entities_list: List[float],
        random_seed: int,
        n_random_trials: int,
        is_pca_version: bool,
        verbose: bool
):
    for filter_relation in filter_relation_list:
        for propensity_score_other_entities in propensity_score_other_entities_list:
            run_single_experiment_setting_of_experiment_known_prop_scores_sar_two_subject_groups(
                filename_ground_truth_dataset=filename_ground_truth_dataset,
                separator_ground_truth_dataset=separator_ground_truth_dataset,
                amie_rule_tsv_filename=amie_rule_tsv_filename,
                experiment_info=KnownPropScoresSARExperimentInfo(
                    dataset_name=dataset_name,
                    target_relation=target_relation,
                    filter_relation=filter_relation,
                    true_prop_scores=PropScoresTwoSARGroups(
                        in_filter=propensity_score_subjects_of_filter_relation,
                        other=propensity_score_other_entities
                    ),
                    is_pca_version=is_pca_version
                ),
                random_seed=random_seed,
                n_random_trials=n_random_trials,
                verbose=verbose,
            )


def run_per_target_and_filter_relation_known_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        dataset_name: str,
        target_relation: str,
        filter_relation: str,
        propensity_score_subjects_of_filter_relation: float,
        propensity_score_other_entities_list: List[float],
        random_seed: int,
        n_random_trials: int,
        is_pca_version: bool,
        verbose: bool
):
    for propensity_score_other_entities in propensity_score_other_entities_list:
        run_single_experiment_setting_of_experiment_known_prop_scores_sar_two_subject_groups(
            filename_ground_truth_dataset=filename_ground_truth_dataset,
            separator_ground_truth_dataset=separator_ground_truth_dataset,
            amie_rule_tsv_filename=amie_rule_tsv_filename,
            experiment_info=KnownPropScoresSARExperimentInfo(
                dataset_name=dataset_name,
                target_relation=target_relation,
                filter_relation=filter_relation,
                true_prop_scores=PropScoresTwoSARGroups(
                    in_filter=propensity_score_subjects_of_filter_relation,
                    other=propensity_score_other_entities
                ),
                is_pca_version=is_pca_version
            ),
            random_seed=random_seed,
            n_random_trials=n_random_trials,
            verbose=verbose
        )

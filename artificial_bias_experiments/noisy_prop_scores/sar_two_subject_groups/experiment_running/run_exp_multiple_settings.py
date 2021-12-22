from typing import List

from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_info import \
    NoisyPropScoresSARExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_core.run_exp_stub import \
    run_single_experiment_setting_of_experiment_noisy_prop_scores_sar_two_subject_groups
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


def run_per_target_relation_experiment_noisy_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,

        dataset_name: str,
        target_relation: str,
        filter_relation_list: List[str],
        true_prop_scores: PropScoresTwoSARGroups,
        noisy_prop_score_in_filter: float,
        noisy_prop_score_not_in_filter_list: List[float],
        is_pca_version: bool,

        random_seed: int,
        n_random_trials: int,
        verbose: bool
):
    for filter_relation in filter_relation_list:
        run_single_experiment_setting_of_experiment_noisy_prop_scores_sar_two_subject_groups(
            filename_ground_truth_dataset=filename_ground_truth_dataset,
            separator_ground_truth_dataset=separator_ground_truth_dataset,
            amie_rule_tsv_filename=amie_rule_tsv_filename,
            experiment_info=NoisyPropScoresSARExperimentInfo(
                dataset_name=dataset_name,
                target_relation=target_relation,
                filter_relation=filter_relation,
                true_prop_scores=true_prop_scores,
                noisy_prop_score_in_filter=noisy_prop_score_in_filter,
                noisy_prop_score_not_in_filter_list=noisy_prop_score_not_in_filter_list,
                is_pca_version=is_pca_version,
            ),
            random_seed=random_seed,
            n_random_trials=n_random_trials,
            verbose=verbose,
        )

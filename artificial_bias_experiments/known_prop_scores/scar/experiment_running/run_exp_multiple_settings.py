from typing import List

from artificial_bias_experiments.known_prop_scores.scar.experiment_core.run_exp_stub import \
    run_single_experiment_setting_of_experiment_known_prop_scores_scar
from artificial_bias_experiments.known_prop_scores.scar.experiment_info import \
    KnownPropScoresSCARExperimentInfo


def run_per_target_relation_experiment_known_prop_scores_scar(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        amie_rule_tsv_filename: str,
        dataset_name: str,
        target_relation: str,
        label_frequency_list: List[float],
        random_seed: int,
        n_random_trials: int,
        is_pca_version: bool,
        verbose: bool
):
    for label_frequency in label_frequency_list:
        run_single_experiment_setting_of_experiment_known_prop_scores_scar(
            filename_ground_truth_dataset=filename_ground_truth_dataset,
            separator_ground_truth_dataset=separator_ground_truth_dataset,
            amie_rule_tsv_filename=amie_rule_tsv_filename,
            experiment_info=KnownPropScoresSCARExperimentInfo(
                dataset_name=dataset_name,
                target_relation=target_relation,
                true_label_frequency=label_frequency,
                is_pca_version=is_pca_version
            ),
            random_seed=random_seed,
            n_random_trials=n_random_trials,
            verbose=verbose
        )

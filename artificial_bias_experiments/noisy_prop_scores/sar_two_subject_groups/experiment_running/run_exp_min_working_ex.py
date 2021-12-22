import os
from typing import List

from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.amie_rule_learning import get_amie_rule_tsv_filename
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_info import \
    NoisyPropScoresSARExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_core.run_exp_stub import \
    run_single_experiment_setting_of_experiment_noisy_prop_scores_sar_two_subject_groups
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


def main():
    dataset_name: str = "yago3_10"
    target_relation = "created"
    filter_relation = "graduatedfrom"

    true_prop_score_in_filter = 1.0
    true_prop_score_other = 0.5
    true_prop_scores = PropScoresTwoSARGroups(
        in_filter=true_prop_score_in_filter,
        other=true_prop_score_other
    )

    noisy_prop_score_in_filter: float = 1.0
    noisy_prop_score_not_in_filter_list: List[float] = [0.1, 0.2, .3, .4, .5, .6, .7, .8, .9, 1]

    random_seed: int = 3
    verbose: bool = True
    is_pca_version: bool = False

    n_random_trials: int = 10

    amie_min_std_confidence: float = 0.1
    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )

    amie_rule_tsv_filename = get_amie_rule_tsv_filename(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        dataset_name=dataset_name,
        min_std_confidence=amie_min_std_confidence
    )

    run_single_experiment_setting_of_experiment_noisy_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        separator_ground_truth_dataset="\t",
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


if __name__ == '__main__':
    main()

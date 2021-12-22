import os
from typing import List

from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_running.run_exp_multiple_settings \
    import run_per_target_and_filter_relation_known_prop_scores_sar_two_subject_groups
from artificial_bias_experiments.amie_rule_learning import \
    get_amie_rule_tsv_filename


def main():
    dataset_name: str = "yago3_10"
    target_relation: str = "created"
    filter_relation: str = "graduatedfrom"

    # target_relation: str = "haschild"
    # filter_relation: str = "ispoliticianof"

    # dataset_name: str = "yago3_10"
    # target_relation: str = "iscitizenof"
    # filter_relation: str = "ispoliticianof"
    propensity_score_subjects_of_filter_relation: float = 0.5
    propensity_score_other_entities_list: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    random_seed: int = 3
    verbose: bool = False
    is_pca_version: bool = True

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

    run_per_target_and_filter_relation_known_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        separator_ground_truth_dataset="\t",
        amie_rule_tsv_filename=amie_rule_tsv_filename,
        dataset_name=dataset_name,
        target_relation=target_relation,
        filter_relation=filter_relation,
        propensity_score_subjects_of_filter_relation=propensity_score_subjects_of_filter_relation,
        propensity_score_other_entities_list=propensity_score_other_entities_list,
        random_seed=random_seed,
        n_random_trials=n_random_trials,
        is_pca_version=is_pca_version,
        verbose=verbose,
    )


if __name__ == '__main__':
    main()

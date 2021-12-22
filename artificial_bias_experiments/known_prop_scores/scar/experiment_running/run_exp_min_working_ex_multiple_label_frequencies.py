import os
from typing import List

from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.amie_rule_learning import \
    get_amie_rule_tsv_filename
from artificial_bias_experiments.known_prop_scores.scar.experiment_running.run_exp_multiple_settings import \
    run_per_target_relation_experiment_known_prop_scores_scar


def main():
    dataset_name: str = "yago3_10"
    target_relation: str = "actedin"

    # dataset_name: str = "yago3_10"
    # target_relation: str = "iscitizenof"
    label_frequency_list: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    random_seed: int = 3
    verbose: bool = True
    is_pca_version: bool = False

    n_random_trials: int = 1

    amie_min_std_confidence: float = 0.1
    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )

    amie_rule_tsv_filename = get_amie_rule_tsv_filename(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        dataset_name=dataset_name,
        min_std_confidence=amie_min_std_confidence
    )

    run_per_target_relation_experiment_known_prop_scores_scar(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        separator_ground_truth_dataset="\t",
        amie_rule_tsv_filename=amie_rule_tsv_filename,
        dataset_name=dataset_name,
        target_relation=target_relation,
        label_frequency_list=label_frequency_list,
        random_seed=random_seed,
        n_random_trials=n_random_trials,
        is_pca_version=is_pca_version,
        verbose=verbose,
    )


if __name__ == '__main__':
    main()

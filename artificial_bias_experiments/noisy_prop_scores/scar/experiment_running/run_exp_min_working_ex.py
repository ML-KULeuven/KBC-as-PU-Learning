import os

from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.amie_rule_learning import \
    get_amie_rule_tsv_filename
from artificial_bias_experiments.noisy_prop_scores.scar.experiment_info import \
    NoisyPropScoresSCARExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.scar.experiment_core.run_exp_stub import \
    run_single_experiment_setting_of_experiment_noisy_prop_scores_scar


def main():
    dataset_name: str = "yago3_10"
    target_relation: str = "haschild"

    # dataset_name: str = "yago3_10"
    # target_relation: str = "iscitizenof"
    true_label_frequency: float = 0.5
    available_label_frequency_list = [0.1, 0.2, .3, .4, .5, .6, .7, .8, .9]
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

    run_single_experiment_setting_of_experiment_noisy_prop_scores_scar(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        separator_ground_truth_dataset="\t",
        amie_rule_tsv_filename=amie_rule_tsv_filename,
        experiment_info=NoisyPropScoresSCARExperimentInfo(
            dataset_name=dataset_name,
            target_relation=target_relation,
            true_label_frequency=true_label_frequency,
            available_label_frequency_list=available_label_frequency_list,
            is_pca_version=is_pca_version,
        ),
        random_seed=random_seed,
        n_random_trials=n_random_trials,
        verbose=verbose,
    )


if __name__ == '__main__':
    main()

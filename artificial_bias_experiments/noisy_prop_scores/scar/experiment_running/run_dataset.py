import os
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd
from dask.delayed import Delayed, delayed
from distributed import Client
from dask_utils.computations import compute_delayed_functions
from dask_utils.dask_initialization import reconnect_client_to_ssh_cluster
from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.amie_rule_learning import get_amie_rule_tsv_filename
from artificial_bias_experiments.known_prop_scores.dataset_generation_file_naming import \
    get_root_dir_experiment_known_propensity_scores
from artificial_bias_experiments.noisy_prop_scores.scar.experiment_info import \
    NoisyPropScoresSCARExperimentInfo
from artificial_bias_experiments.noisy_prop_scores.scar.experiment_core.run_exp_stub import \
    run_single_experiment_setting_of_experiment_noisy_prop_scores_scar
from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth


def run_noisy_prop_scores_scar_both_pca_and_non_pca_for_dataset(dataset_name: str, dask_scheduler_host: str) -> None:

    true_label_frequency_list: List[float] = [0.3, 0.7]
    # true_label_frequency: float = 0.5
    available_label_frequency_list: List[float] = list(
        [0.1, 0.2, .3, .4, .5, .6, .7, .8, .9, 1]
    )
    random_seed: int = 3
    verbose = False
    use_pca_list: List[bool] = [
        False,
        # True
    ]

    n_random_trials: int = 10

    amie_min_std_confidence: float = 0.1
    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )
    separator_ground_truth_dataset = "\t"
    df_ground_truth: pd.DataFrame = get_df_ground_truth(filename_ground_truth_dataset, separator_ground_truth_dataset)
    target_relation_list: List[str] = list(sorted(df_ground_truth["Rel"].unique()))

    amie_rule_tsv_filename = get_amie_rule_tsv_filename(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        dataset_name=dataset_name,
        min_std_confidence=amie_min_std_confidence
    )

    use_dask: bool = True
    list_of_computations: List[Tuple[Delayed, Dict]] = []
    if use_dask:
        scheduler_host: str = dask_scheduler_host
        client: Optional[Client] = reconnect_client_to_ssh_cluster(scheduler_host)
    else:
        client = None

    for use_pca in use_pca_list:
        for true_label_frequency in true_label_frequency_list:
            for target_relation in target_relation_list:
                if use_dask:
                    func_args: Dict[str, Any] = dict(
                        filename_ground_truth_dataset=filename_ground_truth_dataset,
                        separator_ground_truth_dataset=separator_ground_truth_dataset,
                        amie_rule_tsv_filename=amie_rule_tsv_filename,
                        experiment_info=NoisyPropScoresSCARExperimentInfo(
                            dataset_name=dataset_name,
                            target_relation=target_relation,
                            true_label_frequency=true_label_frequency,
                            available_label_frequency_list=available_label_frequency_list,
                            is_pca_version=use_pca,
                        ),
                        random_seed=random_seed,
                        n_random_trials=n_random_trials,
                        verbose=verbose,
                    )
                    delayed_func = delayed(run_single_experiment_setting_of_experiment_noisy_prop_scores_scar)(
                        **func_args
                    )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    run_single_experiment_setting_of_experiment_noisy_prop_scores_scar(
                        filename_ground_truth_dataset=filename_ground_truth_dataset,
                        separator_ground_truth_dataset=separator_ground_truth_dataset,
                        amie_rule_tsv_filename=amie_rule_tsv_filename,
                        experiment_info=NoisyPropScoresSCARExperimentInfo(
                            dataset_name=dataset_name,
                            target_relation=target_relation,
                            true_label_frequency=true_label_frequency,
                            available_label_frequency_list=available_label_frequency_list,
                            is_pca_version=use_pca,
                        ),
                        random_seed=random_seed,
                        n_random_trials=n_random_trials,
                        verbose=verbose,
                    )
    if use_dask:

        dir_log_file: str = os.path.join(
            get_root_dir_experiment_known_propensity_scores(),
            'scar',
            dataset_name
        )
        if not os.path.exists(dir_log_file):
            os.makedirs(dir_log_file)

        logger_name: str = 'ERROR_LOGGER_known_prop_scores_scar_for_' + dataset_name
        logger_file_name: str = os.path.join(
            dir_log_file,
            logger_name
        )

        compute_delayed_functions(
            list_of_computations=list_of_computations,
            client=client,
            nb_of_retries_if_erred=5,
            error_logger_name=logger_name,
            error_logger_file_name=logger_file_name,
        )



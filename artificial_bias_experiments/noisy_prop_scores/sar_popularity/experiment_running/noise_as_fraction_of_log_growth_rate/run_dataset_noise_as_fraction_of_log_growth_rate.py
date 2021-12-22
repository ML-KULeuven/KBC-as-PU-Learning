import os
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd
from dask.delayed import Delayed, delayed
from distributed import Client

from artificial_bias_experiments.amie_rule_learning import get_amie_rule_tsv_filename
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.experiment_running.run_exp_multiple_settings import \
    run_per_log_growth_rate_noisy_prop_scores_sar_popularity
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noise_definition import \
    NoiseTypeSARPopularity
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.noisy_prop_scores_sar_popularity_file_naming import \
    NoisyPropScoresSARPopularityFileNamer

from dask_utils.computations import compute_delayed_functions
from dask_utils.dask_initialization import reconnect_client_to_ssh_cluster

from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from kbc_pul.project_info import data_dir as kbc_pul_data_dir


def run_known_prop_scores_sar_popularity_noise_as_fraction_of_log_growth_rate_for_dataset(dataset_name: str, dask_scheduler_host: str) -> None:

    random_seed: int = 3
    verbose = False
    use_pca_list: List[bool] = [
        False,
        # True
    ]
    log_growth_rate_list: List[float] = [0.01, 0.05, 0.1, 0.5,  1]
    noise_type: NoiseTypeSARPopularity = NoiseTypeSARPopularity.FRACTION_OF_LOG_GROWTH_RATE
    noise_fractions_for_log_growth_rate_list: List[float] = [1.0, 0.9, 1.1]
    noise_levels: List[float] = noise_fractions_for_log_growth_rate_list

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

    target_relation: str
    for use_pca in use_pca_list:
        for target_relation in target_relation_list:
                if use_dask:
                    func_args: Dict[str, Any] = dict(
                        filename_ground_truth_dataset=filename_ground_truth_dataset,
                        separator_ground_truth_dataset=separator_ground_truth_dataset,
                        amie_rule_tsv_filename=amie_rule_tsv_filename,

                        dataset_name=dataset_name,
                        target_relation=target_relation,
                        log_growth_rate_list=log_growth_rate_list,
                        noise_type=noise_type,
                        noise_levels=noise_levels,
                        random_seed=random_seed,

                        n_random_trials=n_random_trials,
                        is_pca_version=use_pca,

                        verbose=verbose
                    )
                    delayed_func = delayed(run_per_log_growth_rate_noisy_prop_scores_sar_popularity)(
                        **func_args
                    )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    run_per_log_growth_rate_noisy_prop_scores_sar_popularity(
                        filename_ground_truth_dataset=filename_ground_truth_dataset,
                        separator_ground_truth_dataset=separator_ground_truth_dataset,
                        amie_rule_tsv_filename=amie_rule_tsv_filename,

                        dataset_name=dataset_name,
                        target_relation=target_relation,
                        log_growth_rate_list=log_growth_rate_list,
                        noise_type=noise_type,
                        noise_levels=noise_levels,
                        random_seed=random_seed,

                        n_random_trials=n_random_trials,
                        is_pca_version=use_pca,

                        verbose=verbose
                    )

    if use_dask:

        dir_log_file: str = NoisyPropScoresSARPopularityFileNamer.get_filename_log_file_dir(
            dataset_name=dataset_name)
        if not os.path.exists(dir_log_file):
            os.makedirs(dir_log_file)

        logger_name: str = 'ERROR_LOGGER_known_prop_scores_sar_for_' + dataset_name
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

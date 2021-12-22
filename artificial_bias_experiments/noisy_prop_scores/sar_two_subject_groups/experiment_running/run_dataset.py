import os
from typing import List, Tuple, Dict, Optional, Any

from dask.delayed import Delayed, delayed
from distributed import Client

from artificial_bias_experiments.evaluation.sar_group_finding_relation_overlap import \
    get_target_relation_to_filter_relation_list_map_and_create_if_non_existent
from dask_utils.computations import compute_delayed_functions
from dask_utils.dask_initialization import reconnect_client_to_ssh_cluster
from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.amie_rule_learning import get_amie_rule_tsv_filename

from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_running.run_exp_multiple_settings import \
    run_per_target_relation_experiment_noisy_prop_scores_sar_two_subject_groups
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    TargetFilterOverlapSettings
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.noisy_prop_scores_sar_two_groups_file_naming import \
    NoisyPropScoresSARTwoGroupsFileNamer
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups


def run_noisy_prop_scores_sar_two_groups_both_pca_and_non_pca_for_dataset(
        dataset_name: str, dask_scheduler_host: str) -> None:

    # true_prop_score_in_filter = 1.0
    # true_prop_score_other = 0.5

    true_prop_score_in_filter = 0.5
    true_prop_score_other_list = [0.3, .7]

    # true_prop_scores = PropScoresTwoSARGroups(
    #     in_filter=true_prop_score_in_filter,
    #     other=true_prop_score_other
    # )

    noisy_prop_score_in_filter: float = true_prop_score_in_filter
    noisy_prop_score_not_in_filter_list: List[float] = [0.1, 0.2, .3, .4, .5, .6, .7, .8, .9, 1]

    random_seed: int = 3
    verbose = False
    use_pca_list: List[bool] = [False, True]

    target_filter_overlap_settings = TargetFilterOverlapSettings(
        fraction_lower_bound=0.1,
        fraction_upper_bound=0.9,
        intersection_absolute_lower_bound=10
    )

    n_random_trials: int = 10

    amie_min_std_confidence: float = 0.1
    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )
    separator_ground_truth_dataset = "\t"
    # df_ground_truth: pd.DataFrame = get_df_ground_truth(filename_ground_truth_dataset, separator_ground_truth_dataset)
    # target_relation_list: List[str] = list(sorted(df_ground_truth["Rel"].unique()))

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
        target_to_filter_relation_list_map: Dict[
            str,
            List[str]
        ] = get_target_relation_to_filter_relation_list_map_and_create_if_non_existent(
            filename_ground_truth_dataset=filename_ground_truth_dataset,
            separator_ground_truth_dataset=separator_ground_truth_dataset,
            dataset_name=dataset_name,
            target_filter_overlap_settings=target_filter_overlap_settings,
            is_pca_version=use_pca
        )

        for true_prop_score_other in true_prop_score_other_list:
            true_prop_scores = PropScoresTwoSARGroups(
                in_filter=true_prop_score_in_filter,
                other=true_prop_score_other
            )

            target_relation: str
            filter_relation_list: List[str]
            for target_relation, filter_relation_list in target_to_filter_relation_list_map.items():
                if use_dask:
                    func_args: Dict[str, Any] = dict(
                        filename_ground_truth_dataset=filename_ground_truth_dataset,
                        separator_ground_truth_dataset="\t",
                        amie_rule_tsv_filename=amie_rule_tsv_filename,

                        dataset_name=dataset_name,
                        target_relation=target_relation,
                        filter_relation_list=filter_relation_list,
                        true_prop_scores=true_prop_scores,
                        noisy_prop_score_in_filter=noisy_prop_score_in_filter,
                        noisy_prop_score_not_in_filter_list=noisy_prop_score_not_in_filter_list,
                        is_pca_version=use_pca,

                        random_seed=random_seed,
                        n_random_trials=n_random_trials,
                        verbose=verbose,
                    )
                    delayed_func = delayed(run_per_target_relation_experiment_noisy_prop_scores_sar_two_subject_groups)(
                        **func_args
                    )
                    list_of_computations.append((delayed_func, func_args))
                else:
                    run_per_target_relation_experiment_noisy_prop_scores_sar_two_subject_groups(
                        filename_ground_truth_dataset=filename_ground_truth_dataset,
                        separator_ground_truth_dataset="\t",
                        amie_rule_tsv_filename=amie_rule_tsv_filename,

                        dataset_name=dataset_name,
                        target_relation=target_relation,
                        filter_relation_list=filter_relation_list,
                        true_prop_scores=true_prop_scores,
                        noisy_prop_score_in_filter=noisy_prop_score_in_filter,
                        noisy_prop_score_not_in_filter_list=noisy_prop_score_not_in_filter_list,
                        is_pca_version=use_pca,

                        random_seed=random_seed,
                        n_random_trials=n_random_trials,
                        verbose=verbose,
                    )
    if use_dask:

        dir_log_file: str = NoisyPropScoresSARTwoGroupsFileNamer.get_filename_log_file_dir(
            dataset_name=dataset_name
        )
        if not os.path.exists(dir_log_file):
            os.makedirs(dir_log_file)

        logger_name: str = 'ERROR_LOGGER_noisy_prop_scores_sar_for_' + dataset_name
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

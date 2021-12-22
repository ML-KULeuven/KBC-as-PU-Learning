import os
from typing import List, Tuple, Dict, Optional, Any

from dask.delayed import Delayed, delayed
from distributed import Client

from dask_utils.computations import compute_delayed_functions
from dask_utils.dask_initialization import reconnect_client_to_ssh_cluster

from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.evaluation.sar_group_finding_relation_overlap import \
    get_target_relation_to_filter_relation_list_map_and_create_if_non_existent
from artificial_bias_experiments.amie_rule_learning import get_amie_rule_tsv_filename
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    TargetFilterOverlapSettings
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_running.run_exp_multiple_settings import \
    run_per_target_relation_experiment_known_prop_scores_sar_two_subject_groups


def run_known_prop_scores_sar_both_pca_and_non_pca_for_dataset(dataset_name: str, dask_scheduler_host: str) -> None:
    propensity_score_subjects_of_filter_relation: float = 0.5
    # propensity_score_other_entities_list: List[float] = [0.2, 0.4, 0.6, 0.8, 1]
    propensity_score_other_entities_list: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    random_seed: int = 3
    verbose = True
    use_pca_list: List[bool] = [
        False,
        True
    ]

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

    amie_rule_tsv_filename = get_amie_rule_tsv_filename(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        dataset_name=dataset_name,
        min_std_confidence=amie_min_std_confidence
    )

    use_dask: bool = False
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

        target_relation: str
        filter_relation_list: List[str]
        for target_relation, filter_relation_list in target_to_filter_relation_list_map.items():
            # if target_relation != 'graduatedfrom':
            #     continue

            if use_dask:
                func_args: Dict[str, Any] = dict(
                    filename_ground_truth_dataset=filename_ground_truth_dataset,
                    separator_ground_truth_dataset=separator_ground_truth_dataset,
                    amie_rule_tsv_filename=amie_rule_tsv_filename,

                    dataset_name=dataset_name,
                    target_relation=target_relation,
                    filter_relation_list=filter_relation_list,

                    propensity_score_subjects_of_filter_relation=propensity_score_subjects_of_filter_relation,
                    propensity_score_other_entities_list=propensity_score_other_entities_list,

                    random_seed=random_seed,
                    n_random_trials=n_random_trials,
                    is_pca_version=use_pca,

                    verbose=verbose
                )
                delayed_func = delayed(run_per_target_relation_experiment_known_prop_scores_sar_two_subject_groups)(
                    **func_args
                )
                list_of_computations.append((delayed_func, func_args))
            else:
                run_per_target_relation_experiment_known_prop_scores_sar_two_subject_groups(
                    filename_ground_truth_dataset=filename_ground_truth_dataset,
                    separator_ground_truth_dataset=separator_ground_truth_dataset,
                    amie_rule_tsv_filename=amie_rule_tsv_filename,

                    dataset_name=dataset_name,
                    target_relation=target_relation,
                    filter_relation_list=filter_relation_list,

                    propensity_score_subjects_of_filter_relation=propensity_score_subjects_of_filter_relation,
                    propensity_score_other_entities_list=propensity_score_other_entities_list,

                    random_seed=random_seed,
                    n_random_trials=n_random_trials,
                    is_pca_version=use_pca,

                    verbose=verbose
                )

    if use_dask:

        dir_log_file: str = KnownPropScoresSARTwoGroupsFileNamer.get_filename_log_file_dir(
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

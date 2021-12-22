import os
from pathlib import Path
from typing import Set

import pandas as pd
from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.known_prop_scores.dataset_generation_file_naming import \
    get_root_dir_experiment_known_propensity_scores
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.pca_selection.confidence_evolution_pca_selection_mechanism import \
    pca_selection_mechanism_known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.known_prop_scores_sar_generate_images import \
    _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from kbc_pul.confidence_naming import ConfidenceEnum


def generate_conf_evolution_pca_selection_images_for_yago3_10():
    dataset_name: str = "yago3_10"
    is_pca_version: bool = True
    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )
    true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O
    conf_estimators_to_ignore: Set[ConfidenceEnum] = set()

    separator_ground_truth_dataset = "\t"
    # df_ground_truth: pd.DataFrame = get_df_ground_truth(filename_ground_truth_dataset, separator_ground_truth_dataset)
    # target_relation_list: List[str] = list(sorted(df_ground_truth["Rel"].unique()))
    scar_propensity_score = 0.5

    # target_filter_overlap_settings = TargetFilterOverlapSettings(
    #     fraction_lower_bound=0.1,
    #     fraction_upper_bound=0.9,
    #     intersection_absolute_lower_bound=10
    # )

    # target_to_filter_relation_list_map: Dict[
    #     str,
    #     List[str]
    # ] = get_target_relation_to_filter_relation_list_map_and_create_if_non_existent(
    #     filename_ground_truth_dataset=filename_ground_truth_dataset,
    #     separator_ground_truth_dataset=separator_ground_truth_dataset,
    #     dataset_name=dataset_name,
    #     target_filter_overlap_settings=target_filter_overlap_settings,
    #     is_pca_version=use_pca
    # )
    root_experiment_dir: str = os.path.join(
        get_root_dir_experiment_known_propensity_scores(),
        'sar_two_subject_groups',
        dataset_name

    )
    path_root_experiment_dir = Path(root_experiment_dir)
    for target_rel_path in path_root_experiment_dir.iterdir():
        if target_rel_path.is_dir():
            for filter_dir in target_rel_path.iterdir():
                if filter_dir.is_dir():
                    target_relation = target_rel_path.name
                    filter_relation = filter_dir.name
                    print(target_relation, filter_relation)
                    try:
                        root_dir_experiment_settings: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_experiment_high_level(
                            dataset_name=dataset_name,
                            target_relation=target_relation,
                            filter_relation=filter_relation,
                            is_pca_version=is_pca_version
                        )

                        image_dir: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_images(
                            use_pca=is_pca_version, dataset_name=dataset_name,
                            scar_propensity_score=scar_propensity_score
                        )
                        if not os.path.exists(image_dir):
                            os.makedirs(image_dir)

                        df_rule_wrappers: pd.DataFrame = _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups(
                            root_dir_experiment_settings=root_dir_experiment_settings,
                            target_relation=target_relation,
                            filter_relation=filter_relation,
                            filter_group_prop_score=scar_propensity_score
                        )

                        filename_root: str = f"known_prop_scores_sar" \
                                             f"_{target_relation}" \
                                             f"_{filter_relation}" \
                                             f"_{is_pca_version}"
                        pca_selection_mechanism_known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule(
                            df_rule_wrappers=df_rule_wrappers,
                            filter_relation=filter_relation,
                            image_dir=image_dir,
                            filename_root=filename_root,
                            scar_propensity_score=scar_propensity_score
                        )

                    except Exception as err:
                        print(f"Problems with {target_relation} images generation")
                        print(err)
                        print()


if __name__ == '__main__':
    generate_conf_evolution_pca_selection_images_for_yago3_10()

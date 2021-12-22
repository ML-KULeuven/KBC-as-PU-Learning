import os
from pathlib import Path
from typing import List

import pandas as pd

from artificial_bias_experiments.images_paper_joint.known_prop_scores_pca_conf.pca_conf_joint_images import \
    generate_pca_conf_image_scar_and_sar_two_groups
from artificial_bias_experiments.known_prop_scores.dataset_generation_file_naming import \
    get_root_dir_experiment_known_propensity_scores
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.known_prop_scores_sar_generate_images import \
    _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from artificial_bias_experiments.known_prop_scores.scar.image_generation.known_prop_scores_scar_generate_images import \
    _get_rule_wrappers_as_dataframe
from artificial_bias_experiments.known_prop_scores.scar.known_prop_scores_scar_file_naming import \
    KnownPropScoresSCARConstantLabelFreqFileNamer
from kbc_pul.project_info import project_dir as kbc_e_metrics_project_dir


def generate_images_for_yago3_10():
    dataset_name: str = "yago3_10"
    use_pca_list: List[bool] = [False]
    # filename_ground_truth_dataset: str = os.path.join(
    #     kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    # )
    # separator_ground_truth_dataset = "\t"
    # df_ground_truth: pd.DataFrame = get_df_ground_truth(filename_ground_truth_dataset, separator_ground_truth_dataset)
    # target_relation_list: List[str] = list(sorted(df_ground_truth["Rel"].unique()))

    scar_propensity_score = 0.5

    root_experiment_dir: str = os.path.join(
        get_root_dir_experiment_known_propensity_scores(),
        'sar_two_subject_groups',
        dataset_name

    )
    image_dir: str = os.path.join(
        kbc_e_metrics_project_dir,
        "images",
        'joint_images',
        "known_prop_scores_cwa_conf"
    )
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    path_root_experiment_dir = Path(root_experiment_dir)
    for target_rel_path in sorted(path_root_experiment_dir.iterdir()):
        if target_rel_path.is_dir():
            for filter_dir in target_rel_path.iterdir():
                if filter_dir.is_dir():
                    target_relation = target_rel_path.name
                    filter_relation = filter_dir.name
                    for is_pca_version in use_pca_list:
                        try:
                            print(f"Generating images for {dataset_name} - {target_relation} (known prop scores, CWA evolution)")
                            # if is_pca_version:
                            #     pca_indicator: str = "pca_version"
                            # else:
                            #     pca_indicator: str = "not_pca"

                            root_dir_experiment_settings_scar: str = KnownPropScoresSCARConstantLabelFreqFileNamer.get_dir_experiment_high_level(
                                dataset_name=dataset_name,
                                target_relation=target_relation,
                                is_pca_version=is_pca_version
                            )

                            df_rule_wrappers_scar: pd.DataFrame = _get_rule_wrappers_as_dataframe(
                                root_dir_experiment_settings=root_dir_experiment_settings_scar,
                                target_relation=target_relation
                            )

                            root_dir_experiment_settings_sar_two_groups: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_experiment_high_level(
                                dataset_name=dataset_name,
                                target_relation=target_relation,
                                filter_relation=filter_relation,
                                is_pca_version=is_pca_version
                            )
                            df_rule_wrappers_sar_two_groups: pd.DataFrame = _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups(
                                root_dir_experiment_settings=root_dir_experiment_settings_sar_two_groups,
                                target_relation=target_relation,
                                filter_relation=filter_relation,
                                filter_group_prop_score=scar_propensity_score
                            )
                            generate_pca_conf_image_scar_and_sar_two_groups(
                                df_rule_wrappers_scar=df_rule_wrappers_scar,
                                df_rule_wrappers_sar_two_groups=df_rule_wrappers_sar_two_groups,
                                target_relation=target_relation,
                                filter_relation_sar_two_groups=filter_relation,
                                scar_propensity_score_sar_two_groups=scar_propensity_score,
                                image_dir=image_dir
                            )

                        except Exception as err:
                            print(f"Problems with {target_relation} {filter_relation} {scar_propensity_score} images generation")
                            print(err)
                            print()


if __name__ == '__main__':
    generate_images_for_yago3_10()

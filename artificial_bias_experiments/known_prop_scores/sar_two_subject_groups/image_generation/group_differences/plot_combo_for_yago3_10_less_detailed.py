import os
from typing import List, Dict

from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.evaluation.sar_group_finding_relation_overlap import \
    get_target_relation_to_filter_relation_list_map_and_create_if_non_existent
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    TargetFilterOverlapSettings
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.plot_combined_group_difference_and_pca_selection_conf_evolution_less_detailed import \
    plot_combined_group_info_known_prop_scores_with_pca_selection_confidence_evolution_less_detailed


def generate_conf_evolution_pca_selection_images_for_yago3_10():
    dataset_name: str = "yago3_10"
    propensity_score_subjects_of_filter_relation: float = 0.5
    use_pca: bool = True

    propensity_score_other_entities_list: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )
    separator_ground_truth_dataset = "\t"

    target_filter_overlap_settings = TargetFilterOverlapSettings(
        fraction_lower_bound=0.1,
        fraction_upper_bound=0.9,
        intersection_absolute_lower_bound=10
    )

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
        for filter_relation in filter_relation_list:
            try:
                plot_combined_group_info_known_prop_scores_with_pca_selection_confidence_evolution_less_detailed(
                    dataset_name=dataset_name,
                    target_relation=target_relation,
                    filter_relation=filter_relation,
                    propensity_score_subjects_of_filter_relation=propensity_score_subjects_of_filter_relation,
                    propensity_score_other_entities_list=propensity_score_other_entities_list
                )
            except Exception as err:
                print(f"Problems with {target_relation} - {filter_relation} image generation")
                print(err)
                print()


if __name__ == '__main__':
    generate_conf_evolution_pca_selection_images_for_yago3_10()

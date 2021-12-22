import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.known_prop_scores.dataset_generation_file_naming import \
    get_root_dir_experiment_known_propensity_scores
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    TargetFilterOverlapSettings, KnownPropScoresSARExperimentInfo
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth
from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups

TargetRelation = str
FilterRelation = str


def get_filename_target_relation_to_subject_relation_mapping(
        dataset_name: str,
        target_filter_overlap_settings: TargetFilterOverlapSettings,
        is_pca_version: bool,
        create_dirs: bool
) -> str:
    if is_pca_version:
        pca_token: str = '_pca_version'
    else:
        pca_token = '_not_pca'
    root_dir_prop_score_grid_search_experiment: str = get_root_dir_experiment_known_propensity_scores()
    dir_dataset: str = os.path.join(
        root_dir_prop_score_grid_search_experiment,
        dataset_name
    )
    if create_dirs and not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset)
    filename_target_relation_to_subject_relation_mapping: str = os.path.join(
        dir_dataset,
        f"target_to_filter_relation_map" +
        pca_token
        + f"_l{target_filter_overlap_settings.fraction_lower_bound}"
          f"_u{target_filter_overlap_settings.fraction_upper_bound}"
          f"_abs_min{target_filter_overlap_settings.intersection_absolute_lower_bound}.json"

    )
    return filename_target_relation_to_subject_relation_mapping


def write_json_target_relation_to_subject_relation_mapping(
        filename_target_relation_to_subject_relation_mapping: str,
        target_to_filter_relation_list_map: Dict[TargetRelation, List[FilterRelation]]
) -> None:
    pretty_frozen = json.dumps(
        target_to_filter_relation_list_map, indent=2, sort_keys=True
    )
    with open(filename_target_relation_to_subject_relation_mapping, 'w') \
            as output_file_target_relation_to_subject_relation_mapping:
        output_file_target_relation_to_subject_relation_mapping.write(pretty_frozen)


def read_json_target_relation_to_subject_relation_mapping(
        filename_target_relation_to_subject_relation_mapping: str
) -> Dict[TargetRelation, List[FilterRelation]]:
    with open(filename_target_relation_to_subject_relation_mapping, 'r') as input_file:
        target_to_filter_relation_list_map: Dict[TargetRelation, List[FilterRelation]] = json.load(input_file)
    return target_to_filter_relation_list_map


def update_map(
        mapping: Dict[TargetRelation, List[FilterRelation]],
        target_relation: TargetRelation,
        filter_relation: FilterRelation
) -> None:
    o_filter_relation_list: Optional[List[FilterRelation]] = mapping.get(target_relation, None)
    if o_filter_relation_list is None:
        mapping[target_relation] = [filter_relation]
    else:
        o_filter_relation_list.append(filter_relation)


def create_target_to_filter_relation_map(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        dataset_name: str,
        target_filter_overlap_settings: TargetFilterOverlapSettings,
        is_pca_version: bool
) -> None:
    df_ground_truth: pd.DataFrame = get_df_ground_truth(filename_ground_truth_dataset, separator_ground_truth_dataset)
    all_relations_list: List[str] = list(df_ground_truth["Rel"].unique())

    target_to_filter_relation_list_map: Dict[TargetRelation, List[FilterRelation]] = dict()
    for relation_index, current_relation in tqdm(enumerate(all_relations_list), total=len(all_relations_list)):

        relation_ent1_set: np.ndarray = df_ground_truth[
            df_ground_truth["Rel"] == current_relation
            ]["Subject"].unique()

        n_subject_ents_current_relation: int = len(relation_ent1_set)

        for other_relation_name in all_relations_list[relation_index + 1:]:
            other_relation_ent1_set: pd.DataFrame = df_ground_truth[
                df_ground_truth["Rel"] == other_relation_name
                ]["Subject"].unique()

            n_subject_ents_other_relation: int = len(other_relation_ent1_set)
            ent_set_intersection = np.intersect1d(relation_ent1_set, other_relation_ent1_set, assume_unique=True)

            n_ents_intersection: int = len(ent_set_intersection)
            if n_ents_intersection > target_filter_overlap_settings.intersection_absolute_lower_bound:
                fraction_rel1_ents_filtered_by_rel2: float = n_ents_intersection / n_subject_ents_current_relation

                fraction_rel2_ents_filtered_by_rel1: float = n_ents_intersection / n_subject_ents_other_relation

                if target_filter_overlap_settings.is_strictly_between_fraction_bounds(
                        fraction_rel1_ents_filtered_by_rel2
                ):
                    # use R1 as target, R2 as filter
                    update_map(target_to_filter_relation_list_map,
                               target_relation=current_relation,
                               filter_relation=other_relation_name
                               )

                if target_filter_overlap_settings.is_strictly_between_fraction_bounds(
                        fraction_rel2_ents_filtered_by_rel1
                ):
                    # use R1 as target, R2 as filter
                    update_map(target_to_filter_relation_list_map,
                               target_relation=other_relation_name,
                               filter_relation=current_relation
                               )
    # write out to file
    filename_target_relation_to_subject_relation_mapping = get_filename_target_relation_to_subject_relation_mapping(
        dataset_name=dataset_name,
        target_filter_overlap_settings=target_filter_overlap_settings,
        is_pca_version=is_pca_version,
        create_dirs=True
    )
    write_json_target_relation_to_subject_relation_mapping(
        filename_target_relation_to_subject_relation_mapping=filename_target_relation_to_subject_relation_mapping,
        target_to_filter_relation_list_map=target_to_filter_relation_list_map
    )


def only_keep_for_which_not_all_exist(
        dataset_name: str,
        target_to_filter_relation_list_map: Dict[TargetRelation, List[FilterRelation]],
        propensity_score_subjects_of_filter_relation: float,
        propensity_score_other_entities_list: List[float],
        is_pca_version: bool
):
    new_target_to_filter_relation_list_map: Dict[TargetRelation, List[FilterRelation]] = dict()

    target_relation: TargetRelation
    filter_relation_list: List[FilterRelation]
    for target_relation, filter_relation_list in target_to_filter_relation_list_map.items():
        for filter_relation in filter_relation_list:
            propensity_score_other_entities: float
            for propensity_score_other_entities in propensity_score_other_entities_list:
                dir_rule_wrappers: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_experiment_specific(
                    experiment_info=KnownPropScoresSARExperimentInfo(
                        dataset_name=dataset_name,
                        target_relation=target_relation,
                        filter_relation=filter_relation,
                        true_prop_scores=PropScoresTwoSARGroups(
                            in_filter=propensity_score_subjects_of_filter_relation,
                            other=propensity_score_other_entities
                        ),
                        is_pca_version=is_pca_version
                    ),
                )
                path_dir_rule_wrappers: Path = Path(dir_rule_wrappers)
                count_rule_wrappers: int = 0
                for path_rule_wrapper in path_dir_rule_wrappers.iterdir():
                    if path_rule_wrapper.is_file() and path_rule_wrapper.suffix == '.gz':
                        count_rule_wrappers += 1
                        break
                if count_rule_wrappers == 0:
                    o_filter_relation_list: Optional[List[TargetRelation]
                    ] = new_target_to_filter_relation_list_map.get(target_relation, None)
                    if o_filter_relation_list is None:
                        new_target_to_filter_relation_list_map[target_relation] = [filter_relation]
                    else:
                        if filter_relation in o_filter_relation_list:
                            pass
                        else:
                            o_filter_relation_list.append(filter_relation)
                else:
                    pass
    return new_target_to_filter_relation_list_map


def get_target_relation_to_filter_relation_list_map_and_create_if_non_existent(
        filename_ground_truth_dataset: str,
        separator_ground_truth_dataset: str,
        dataset_name: str,
        target_filter_overlap_settings: TargetFilterOverlapSettings,
        is_pca_version: bool
) -> Dict[
    TargetRelation,
    List[FilterRelation]
]:
    filename_target_relation_to_subject_relation_mapping = get_filename_target_relation_to_subject_relation_mapping(
        dataset_name=dataset_name,
        target_filter_overlap_settings=target_filter_overlap_settings,
        is_pca_version=is_pca_version,
        create_dirs=True
    )
    if not os.path.exists(filename_target_relation_to_subject_relation_mapping):
        print("Creating target_relation to filter_relation mapping")
        create_target_to_filter_relation_map(
            filename_ground_truth_dataset=filename_ground_truth_dataset,
            separator_ground_truth_dataset=separator_ground_truth_dataset,
            dataset_name=dataset_name,
            target_filter_overlap_settings=target_filter_overlap_settings,
            is_pca_version=is_pca_version
        )

    target_to_filter_relation_list_map: Dict[
        TargetRelation,
        List[FilterRelation]
    ] = read_json_target_relation_to_subject_relation_mapping(
        filename_target_relation_to_subject_relation_mapping=filename_target_relation_to_subject_relation_mapping
    )
    return target_to_filter_relation_list_map


def test():
    dataset_name: str = "yago3_10"

    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )
    separator_ground_truth_dataset: str = "\t"

    is_pca_version: bool = False
    fraction_lower_bound: float = 0.1
    fraction_upper_bound: float = 0.9
    intersection_absolute_lower_bound: int = 10

    create_target_to_filter_relation_map(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        separator_ground_truth_dataset=separator_ground_truth_dataset,
        dataset_name=dataset_name,
        fraction_lower_bound=fraction_lower_bound,
        fraction_upper_bound=fraction_upper_bound,
        intersection_absolute_lower_bound=intersection_absolute_lower_bound,
        is_pca_version=False
    )


if __name__ == '__main__':
    test()

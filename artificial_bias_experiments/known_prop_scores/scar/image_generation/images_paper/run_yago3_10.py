import os
from typing import List

import pandas as pd
from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.known_prop_scores.scar.image_generation.known_prop_scores_scar_generate_images import \
    generate_images_conf_comparison_known_prop_scores_scar
from artificial_bias_experiments.known_prop_scores.scar.image_generation.images_paper.known_prop_scores_scar_generate_images import \
    generate_paper_images_conf_comparison_known_prop_scores_scar
from kbc_pul.experiments_utils.load_df_ground_truth import get_df_ground_truth


def generate_images_for_yago3_10():
    dataset_name: str = "yago3_10"
    use_pca_list: List[bool] = [False, True]
    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )
    separator_ground_truth_dataset = "\t"
    df_ground_truth: pd.DataFrame = get_df_ground_truth(filename_ground_truth_dataset, separator_ground_truth_dataset)
    target_relation_list: List[str] = list(sorted(df_ground_truth["Rel"].unique()))

    for use_pca in use_pca_list:
        for target_relation in target_relation_list:
            try:
                print(f"Generating images for {dataset_name} - {target_relation} (known prop scores, SCAR)")
                generate_paper_images_conf_comparison_known_prop_scores_scar(
                    dataset_name="yago3_10",
                    target_relation=target_relation,
                    is_pca_version=use_pca
                )
            except Exception as err:
                print(f"Problems with {target_relation} images generation")
                print(err)
                print()


if __name__ == '__main__':
    generate_images_for_yago3_10()

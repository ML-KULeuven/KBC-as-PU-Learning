import os

from kbc_pul.project_info import data_dir as kbc_pul_data_dir
from kbc_pul.project_info import project_dir as kbc_e_metrics_project_dir


def get_root_dir_experiment_known_propensity_scores() -> str:
    root_dir_prop_score_grid_search_experiment: str = os.path.join(
        kbc_pul_data_dir,
        'artificial_bias_experiments',
        'known_prop_scores'
    )
    return root_dir_prop_score_grid_search_experiment


def get_root_dir_experiment_noisy_propensity_scores() -> str:
    root_dir_prop_score_grid_search_experiment: str = os.path.join(
        kbc_pul_data_dir,
        'artificial_bias_experiments',
        'noisy_prop_scores'
    )
    return root_dir_prop_score_grid_search_experiment


def get_pca_token(use_pca: bool) -> str:
    if use_pca:
        pca_token: str = "pca_version"

    else:
        pca_token: str = "not_pca"
    return pca_token


def get_root_dir_images_known_prop_scores() -> str:
    image_dir: str = os.path.join(
        kbc_e_metrics_project_dir,
        'images',
        'artificial_bias_experiments',
        'known_prop_scores'
    )
    return image_dir


def get_root_dir_images_noisy_prop_scores() -> str:
    image_dir: str = os.path.join(
        kbc_e_metrics_project_dir,
        'images',
        'artificial_bias_experiments',
        'noisy_prop_scores'
    )
    return image_dir

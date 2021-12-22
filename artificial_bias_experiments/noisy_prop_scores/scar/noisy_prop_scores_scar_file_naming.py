import os

from artificial_bias_experiments.known_prop_scores.dataset_generation_file_naming import \
    get_root_dir_experiment_noisy_propensity_scores, get_pca_token, get_root_dir_images_noisy_prop_scores
from artificial_bias_experiments.noisy_prop_scores.scar.experiment_info import \
    NoisyPropScoresSCARExperimentInfo


class NoisyPropScoresSCARFileNamer:

    @staticmethod
    def get_selection_mechanism_token() -> str:
        return "scar"

    @classmethod
    def get_root_dir_experiment(cls) -> str:
        return get_root_dir_experiment_noisy_propensity_scores()

    @classmethod
    def get_root_dir_images(cls) -> str:
        return get_root_dir_images_noisy_prop_scores()

    @classmethod
    def get_dir_experiment_high_level(cls,
                                      dataset_name: str,
                                      target_relation: str,
                                      is_pca_version: bool
                                      ) -> str:
        experiment_dir: str = os.path.join(
            cls.get_root_dir_experiment(),
            cls.get_selection_mechanism_token(),
            dataset_name,
            target_relation,
            get_pca_token(is_pca_version),
        )
        return experiment_dir

    @classmethod
    def get_dir_experiment_specific(cls,
                                    experiment_info: NoisyPropScoresSCARExperimentInfo
                                    ) -> str:
        experiment_dir: str = os.path.join(
            cls.get_dir_experiment_high_level(
                dataset_name=experiment_info.dataset_name,
                target_relation=experiment_info.target_relation,
                is_pca_version=experiment_info.is_pca_version
            ),
            f"c{experiment_info.true_label_frequency}"
        )
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        return experiment_dir

    @classmethod
    def get_dir_images(cls, use_pca: bool, dataset_name: str, true_label_frequency) -> str:
        image_dir: str = os.path.join(
            cls.get_root_dir_images(),
            cls.get_selection_mechanism_token(),
            get_pca_token(use_pca),
            dataset_name,
            f"c{true_label_frequency}"
        )
        return image_dir

    @classmethod
    def get_filename_log_file_dir(cls, dataset_name: str) -> str:
        dir_log_file: str = os.path.join(
            cls.get_root_dir_experiment(),
            cls.get_selection_mechanism_token(),
            dataset_name
        )
        return dir_log_file
#
# def get_dir_experiment_noisy_propensity_scores_scar(
#         experiment_info: NoisyPropScoresSCARExperimentInfo,
# ) -> str:
#     if experiment_info.is_pca_version:
#         pca_token: str = "pca_version"
#     else:
#         pca_token: str = "not_pca"
#
#     experiment_dir: str = os.path.join(
#         get_root_dir_experiment_noisy_propensity_scores(),
#         'scar',
#         experiment_info.dataset_name,
#         experiment_info.target_relation,
#         pca_token,
#         f"c{experiment_info.true_label_frequency}"
#     )
#     if not os.path.exists(experiment_dir):
#         os.makedirs(experiment_dir)
#     return experiment_dir
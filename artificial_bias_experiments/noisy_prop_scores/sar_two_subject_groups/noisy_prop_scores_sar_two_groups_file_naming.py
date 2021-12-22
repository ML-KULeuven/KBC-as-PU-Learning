import os

from artificial_bias_experiments.known_prop_scores.dataset_generation_file_naming import \
    get_root_dir_experiment_noisy_propensity_scores, get_pca_token, get_root_dir_images_noisy_prop_scores
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_info import \
    NoisyPropScoresSARExperimentInfo


class NoisyPropScoresSARTwoGroupsFileNamer:
    @staticmethod
    def get_selection_mechanism_token() -> str:
        return "sar_two_subject_groups"

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
                                      filter_relation: str,
                                      is_pca_version: bool
                                      ) -> str:
        experiment_dir: str = os.path.join(
            cls.get_root_dir_experiment(),
            cls.get_selection_mechanism_token(),
            dataset_name,
            target_relation,
            filter_relation,
            get_pca_token(is_pca_version),
        )
        return experiment_dir

    @classmethod
    def get_dir_experiment_specific(cls,
                                    experiment_info: NoisyPropScoresSARExperimentInfo
                                    ) -> str:
        experiment_dir: str = os.path.join(
            cls.get_dir_experiment_high_level(
                dataset_name=experiment_info.dataset_name,
                target_relation=experiment_info.target_relation,
                filter_relation=experiment_info.filter_relation,
                is_pca_version=experiment_info.is_pca_version
            ),
            f"s_prop{experiment_info.true_prop_scores.in_filter}"
            f"_ns_prop{experiment_info.true_prop_scores.other}"
        )
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        return experiment_dir

    @classmethod
    def get_dir_images(cls, use_pca: bool, dataset_name: str,
                       true_prop_score_in_filter: float,
                       true_prop_score_other: float
                       ) -> str:
        image_dir: str = os.path.join(
            cls.get_root_dir_images(),
            cls.get_selection_mechanism_token(),
            get_pca_token(use_pca),
            dataset_name,
            f"s_prop{true_prop_score_in_filter}"
            f"_ns_prop{true_prop_score_other}"
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

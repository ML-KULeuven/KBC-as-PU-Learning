import os

from artificial_bias_experiments.known_prop_scores.dataset_generation_file_naming import \
    get_root_dir_experiment_noisy_propensity_scores, get_root_dir_images_known_prop_scores, get_pca_token
from artificial_bias_experiments.noisy_prop_scores.sar_popularity.experiment_info import \
    NoisyPropScoresSARPopularityExperimentInfo


class NoisyPropScoresSARPopularityFileNamer:

    @staticmethod
    def get_selection_mechanism_token() -> str:
        return "sar_popularity"

    @classmethod
    def get_root_dir_experiment(cls) -> str:
        return get_root_dir_experiment_noisy_propensity_scores()

    @classmethod
    def get_root_dir_images(cls) -> str:
        return get_root_dir_images_known_prop_scores()

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
    def get_dir_experiment_specific(cls, experiment_info: NoisyPropScoresSARPopularityExperimentInfo) -> str:
        experiment_dir: str = os.path.join(
            cls.get_dir_experiment_high_level(
                dataset_name=experiment_info.dataset_name,
                target_relation=experiment_info.target_relation,
                is_pca_version=experiment_info.is_pca_version
            ),
            experiment_info.noise_type.value,
            f"log_growth_rate{experiment_info.log_growth_rate}",
        )
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        return experiment_dir

    @classmethod
    def get_dir_images(cls, use_pca: bool, dataset_name: str, ) -> str:
        image_dir: str = os.path.join(
            cls.get_root_dir_images(),
            cls.get_selection_mechanism_token(),
            get_pca_token(use_pca),
            dataset_name
        )
        return image_dir

    @classmethod
    def get_dir_rule_wrappers(cls, experiment_info: NoisyPropScoresSARPopularityExperimentInfo) -> str:
        dir_rule_wrappers: str = os.path.join(
            cls.get_dir_experiment_specific(experiment_info=experiment_info),
            'rule_wrappers'
        )
        return dir_rule_wrappers

    @classmethod
    def get_filename_log_file_dir(cls, dataset_name: str) -> str:
        dir_log_file: str = os.path.join(
            cls.get_root_dir_experiment(),
            cls.get_selection_mechanism_token(),
            dataset_name
        )
        return dir_log_file

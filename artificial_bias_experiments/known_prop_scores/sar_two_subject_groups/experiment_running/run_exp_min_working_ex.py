import os

from kbc_pul.project_info import data_dir as kbc_pul_data_dir

from artificial_bias_experiments.amie_rule_learning import \
    get_amie_rule_tsv_filename
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_info import \
    KnownPropScoresSARExperimentInfo
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.experiment_running.run_exp_multiple_settings import \
    run_single_experiment_setting_of_experiment_known_prop_scores_sar_two_subject_groups

from kbc_pul.observed_data_generation.sar_two_subject_groups.sar_two_subject_groups_prop_scores import \
    PropScoresTwoSARGroups

#
# tmp_dict = dict(
#     filename_ground_truth_dataset='/home/joschout/Projects/VDAB/VDAB-Project/data/yago3_10/cleaned_csv/train.csv',
#     separator_ground_truth_dataset='\t',
#     amie_rule_tsv_filename='/home/joschout/Repos/KBC-e-metrics/data/artificial_bias_experiments/yago3_10/amie/yago3_10_amie_rules_min_std_conf0.1.tsv',
#     dataset_name='yago3_10', target_relation='hascurrency',
#     filter_relation_list=['dealswith', 'exports', 'hascapital', 'hasneighbor', 'hasofficiallanguage', 'imports', 'owns',
#                           'participatedin'], propensity_score_subjects_of_filter_relation=0.8,
#     propensity_score_other_entities_list=[0.2, 0.4, 0.6, 0.8, 1], random_seed=3, n_random_trials=10,
#     is_pca_version=False, verbose=False)


def main():
    dataset_name: str = "yago3_10"
    target_relation: str = "haschild"
    filter_relation: str = "ispoliticianof"
    # target_relation: str = "hascurrency"
    # filter_relation: str = "participatedin"

    propensity_score_subjects_of_filter_relation: float = 0.8
    propensity_score_other_entities: float = 0.2
    random_seed: int = 3
    verbose: bool = True
    is_pca_version: bool = False

    n_random_trials: int = 10

    amie_min_std_confidence: float = 0.1
    filename_ground_truth_dataset: str = os.path.join(
        kbc_pul_data_dir, dataset_name, 'cleaned_csv', 'train.csv'
    )

    amie_rule_tsv_filename = get_amie_rule_tsv_filename(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        dataset_name=dataset_name,
        min_std_confidence=amie_min_std_confidence
    )

    # for filter_relation in tmp_dict['filter_relation_list']:
    #     for propensity_score_other_entities in tmp_dict['propensity_score_other_entities_list']:
    experiment_info = KnownPropScoresSARExperimentInfo(
        dataset_name=dataset_name,
        target_relation=target_relation,
        filter_relation=filter_relation,
        true_prop_scores=PropScoresTwoSARGroups(
            in_filter=propensity_score_subjects_of_filter_relation,
            other=propensity_score_other_entities
        ),
        is_pca_version=is_pca_version
    )

    run_single_experiment_setting_of_experiment_known_prop_scores_sar_two_subject_groups(
        filename_ground_truth_dataset=filename_ground_truth_dataset,
        separator_ground_truth_dataset="\t",
        amie_rule_tsv_filename=amie_rule_tsv_filename,
        experiment_info=experiment_info,
        random_seed=random_seed,
        n_random_trials=n_random_trials,
        verbose=verbose,
    )


if __name__ == '__main__':
    main()

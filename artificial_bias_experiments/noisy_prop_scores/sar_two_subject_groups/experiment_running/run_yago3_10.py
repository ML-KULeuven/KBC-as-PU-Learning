
from artificial_bias_experiments.noisy_prop_scores.sar_two_subject_groups.experiment_running.run_dataset import \
    run_noisy_prop_scores_sar_two_groups_both_pca_and_non_pca_for_dataset


def main():
    run_noisy_prop_scores_sar_two_groups_both_pca_and_non_pca_for_dataset(
        dataset_name="yago3_10",
        dask_scheduler_host="pinac21"
    )


if __name__ == '__main__':
    main()

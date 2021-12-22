from artificial_bias_experiments.known_prop_scores.scar.experiment_running.run_dataset import \
    run_known_prop_scores_scar_both_pca_and_non_pca_for_dataset


def main():
    run_known_prop_scores_scar_both_pca_and_non_pca_for_dataset(
        dataset_name="yago3_10",
        dask_scheduler_host="pinac37"
    )


if __name__ == '__main__':
    main()

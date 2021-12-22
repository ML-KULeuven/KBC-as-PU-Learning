from artificial_bias_experiments.noisy_prop_scores.sar_popularity.experiment_running.noise_as_fraction_of_log_growth_rate.run_dataset_noise_as_fraction_of_log_growth_rate import \
    run_known_prop_scores_sar_popularity_noise_as_fraction_of_log_growth_rate_for_dataset


def main():
    run_known_prop_scores_sar_popularity_noise_as_fraction_of_log_growth_rate_for_dataset(
        dataset_name="yago3_10",
        dask_scheduler_host="pinac21"
    )


if __name__ == '__main__':
    main()

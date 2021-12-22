# Files to run for Experiments

The following is a description on how to run the experiments.

The experiments investigate how the confidence estimators vary i.f.o. varying propensity scores

## The SCAR_p selection mechanism

### Known propensity scores (Q1)
For a minimal working example, see:
[run_exp_min_working_ex.py](../artificial_bias_experiments/known_prop_scores/scar/experiment_running/run_exp_min_working_ex.py)
or 
[run_exp_min_working_ex_multiple_label_frequencies.py](../artificial_bias_experiments/known_prop_scores/scar/experiment_running/run_exp_min_working_ex_multiple_label_frequencies.py)

To run the experiment for all predicates in yago3_10, run:
[run_yago3_10.py](../artificial_bias_experiments/known_prop_scores/scar/experiment_running/run_yago3_10.py)

### Noisy propensity scores (Q2)
For a minimal working example, see:
[run_exp_min_working_ex.py](../artificial_bias_experiments/noisy_prop_scores/scar/experiment_running/run_exp_min_working_ex.py)

To run the experiment for all predicates in yago3_10, run:
[run_yago3_10.py](../artificial_bias_experiments/noisy_prop_scores/scar/experiment_running/run_yago3_10.py)




## The SAR_{group} selection mechanism
### Known propensity scores (Q1)

For a minimal working example, see:
[run_exp_min_working_ex.py](../artificial_bias_experiments/known_prop_scores/sar_two_subject_groups/experiment_running/run_exp_min_working_ex.py)
or
[run_exp_min_working_ex_multiple_prop_scores.py](../artificial_bias_experiments/known_prop_scores/sar_two_subject_groups/experiment_running/run_exp_min_working_ex_multiple_prop_scores.py)

To run the experiments for yago3_10, run:
[run_yago3_10.py](../artificial_bias_experiments/known_prop_scores/sar_two_subject_groups/experiment_running/run_yago3_10.py)

### Noisy propensity scores (Q2 and Q3)
For a minimal working example, see:
[run_exp_min_working_ex.py](../artificial_bias_experiments/noisy_prop_scores/sar_two_subject_groups/experiment_running/run_exp_min_working_ex.py)
To run the experiments for yago3_10, run:
[run_yago3_10.py](../artificial_bias_experiments/noisy_prop_scores/sar_two_subject_groups/experiment_running/run_yago3_10.py)

## The SAR_{popularity} selection mechanism

To run the experiments fo yago3_10, run: 
[run_yago3_10_noise_as_fraction_of_log_growth_rate.py](../artificial_bias_experiments/noisy_prop_scores/sar_popularity/experiment_running/noise_as_fraction_of_log_growth_rate/run_yago3_10_noise_as_fraction_of_log_growth_rate.py)

# Generating the tables in the paper

## Appendix (Table 6) - The non-recursive rules

To obtain the table in the appendix listing all non-recursive rules, run:
[generate_rules_latex_table.ipynb](../notebooks/artificial_bias_experiments/paper_tables/generate_rules_latex_table.ipynb)
which results in the file:
[amie-rules-non-recursive.tex](../paper_latex_tables/amie-rules/amie-rules-non-recursive.tex)


## Appendix (Table 3) - Results for Q1 & Q2 averaged per predicate - SCAR_p

To obtain this table, run:
[noisy_prop_scores_scar.ipynb](../notebooks/artificial_bias_experiments/noisy_prop_scores/scar/table/noisy_prop_scores_scar.ipynb)
to obtain:
[confidence-error-table-scar-rerun-agg-per-p.tex](../paper_latex_tables/known_prop_scores/scar/confidence-error-table-scar-rerun-agg-per-p.tex)

## Appendix (Table 4) - Results for Q1 & Q2 averaged per predicate - SAR_{group}

To obtain this table, run:
[noisy_prop_scores_sar_two_subject_groups_no_pca_table.ipynb](../notebooks/artificial_bias_experiments/noisy_prop_scores/sar_two_subject_groups/table/noisy_prop_scores_sar_two_subject_groups_no_pca_table.ipynb)
to obtain:
[confidence-error-table-sar-two-subject-groups-agg-per-p.tex](../paper_latex_tables/known_prop_scores/sar_two_groups/confidence-error-table-sar-two-subject-groups-agg-per-p.tex)

## Appendix (Table 5)- Results for Q1 & Q2 averaged per predicate - SAR_{popularity}
To obtain this table, run:
[noisy_prop_scores_popularity_no_pca_table.ipynb](../notebooks/artificial_bias_experiments/noisy_prop_scores/sar_popularity/noise_as_fraction_of_log_growth_rate/table/noisy_prop_scores_popularity_no_pca_table.ipynb)
to obtain:
[confidence-error-table-sar-popularity-agg-per-p.tex](./tables-experiments/SAR/confidence-error-table-sar-popularity-agg-per-p.tex)


## Main Paper Table 1 - Results for Q1 & Q2, averaged over all predicates

To obtain this table, run:
[combine_one_line_table_summaries.ipynb](../notebooks/artificial_bias_experiments/paper_tables/combine_one_line_table_summaries.ipynb)
Note that this requires (see tables for appendices):
* [noisy_scar_single_row_summary.tsv](../paper_latex_tables/known_prop_scores/scar/noisy_scar_single_row_summary.tsv) (created in [noisy_prop_scores_scar.ipynb](../notebooks/artificial_bias_experiments/noisy_prop_scores/scar/table/noisy_prop_scores_scar.ipynb).
* [noisy_sar_two_groups_single_row_summary.tsv](../paper_latex_tables/known_prop_scores/sar_two_groups/noisy_sar_two_groups_single_row_summary.tsv) (created in [noisy_prop_scores_sar_two_subject_groups_no_pca_table.ipynb](../notebooks/artificial_bias_experiments/noisy_prop_scores/sar_two_subject_groups/table/noisy_prop_scores_sar_two_subject_groups_no_pca_table.ipynb)).
* [noisy_sar_popularity_single_row_summary.tsv](../paper_latex_tables/known_prop_scores/sar_popularity/noisy_sar_popularity_single_row_summary.tsv) (created in [noisy_prop_scores_popularity_no_pca_table.ipynb](../notebooks/artificial_bias_experiments/noisy_prop_scores/sar_popularity/noise_as_fraction_of_log_growth_rate/table/noisy_prop_scores_popularity_no_pca_table.ipynb))

which results in the file:
[confidence-error-summary-selection-mechanisms.tex](../paper_latex_tables/known_prop_scores/summary_selection_mechanisms/confidence-error-summary-selection-mechanisms.tex)

## Main Paper Table 2 - Aggregated results for Q3
To obtain this table, run:
[noisy_prop_scores_pca_version_with_pca_scenario_rule_filtering_table.ipynb](../notebooks/artificial_bias_experiments/noisy_prop_scores/sar_two_subject_groups/table/noisy_prop_scores_pca_version_with_pca_scenario_rule_filtering_table.ipynb)
which results in the file:
[confidence-error-table-sar-two-subject-groups-pca_version-case-specific_agg-per-p.tex](../paper_latex_tables/known_prop_scores/sar_two_groups/confidence-error-table-sar-two-subject-groups-pca_version-case-specific_agg-per-p.tex)


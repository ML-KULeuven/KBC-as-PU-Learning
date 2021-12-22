import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from artificial_bias_experiments.evaluation.confidence_comparison.df_utils import ColumnNamesInfo, \
    get_df_diffs_between_true_conf_and_confidence_estimators_melted
from artificial_bias_experiments.image_generation.image_generation_utils import \
    get_confidence_difference_label_string
from kbc_pul.confidence_naming import ConfidenceEnum

sns.set(style="whitegrid")


def pca_selection_mechanism_known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule(
        df_rule_wrappers: pd.DataFrame,
        filter_relation: str,
        image_dir: str,
        filename_root: str,
        scar_propensity_score: float

) -> None:
    dir_conf_star_vs_pca_estimators: str = os.path.join(
        image_dir,
        "pca_selection_confidence_evolution_true_conf_star_vs_pca_estimators"
    )
    if not os.path.exists(dir_conf_star_vs_pca_estimators):
        os.makedirs(dir_conf_star_vs_pca_estimators)

    column_names_logistics: List[str] = [
        'target_relation',
        'filter_relation',
        'prop_score_subj',
        'prop_score_other',
        'random_trial_index',
        'Rule',
    ]

    for true_conf in [ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O, ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S]:

        conf_estimators_to_compare = ConfidenceEnum.get_estimators_of(true_conf)
        confs_comparison = [true_conf] + conf_estimators_to_compare
        column_names_confidence_comparison = [
            conf.get_name()
            for conf in confs_comparison
        ]
        column_names_info = ColumnNamesInfo(
            true_conf=true_conf,
            column_name_true_conf=true_conf.get_name(),
            conf_estimators=conf_estimators_to_compare,
            column_names_conf_estimators=[
                col.get_name()
                for col in conf_estimators_to_compare
            ],
            column_names_logistics=column_names_logistics
        )
        color_palette = {
            conf.get_name(): conf.get_hex_color_str()
            for conf in confs_comparison
        }

        df_conf_comp = df_rule_wrappers[
            column_names_logistics + column_names_confidence_comparison
        ]

        n_random_trials = len(df_conf_comp["random_trial_index"].unique())

        rule_name: str
        for rule_name in df_rule_wrappers["Rule"].unique():
            df_conf_comp_single_rule = df_conf_comp[
                df_conf_comp["Rule"] == rule_name
                ]

            df_conf_comp_melted = df_conf_comp_single_rule.melt(
                id_vars=column_names_logistics,
                var_name='confidence_type',
                value_name='confidence_value'
            )

            df_diff_to_true_conf_melted: pd.DataFrame = get_df_diffs_between_true_conf_and_confidence_estimators_melted(
                df_rule_wrappers=df_conf_comp_single_rule,
                column_names_info=column_names_info
            )

            if true_conf == ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O:
                direction_token = "S_to_O"
            elif true_conf == ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S:
                direction_token = "O_to_S"
            else:
                direction_token = "error"

            filename_image_confidence_evolution: str = os.path.join(
                dir_conf_star_vs_pca_estimators,
                f"{filename_root}_conf_star_estimates_{rule_name}_{direction_token}.png"
            )

            figsize = (10, 5)
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            error_metric_label: str = get_confidence_difference_label_string(true_conf)

            conf_max = df_conf_comp_melted['confidence_value'].max()
            conf_min = df_conf_comp_melted['confidence_value'].min()
            conf_scar_y_pos = conf_min + (conf_max-conf_min)/2

            fig: Figure
            axes: Axes
            ax_confs: Axes = sns.lineplot(
                x="prop_score_other",
                y='confidence_value',
                hue="confidence_type",
                # hue_order=order,
                palette=color_palette,
                marker="o",
                data=df_conf_comp_melted,
                ax=axes[0]
            )
            ax_confs.set_xlim([0.0, 1.0])
            # ax_confs.set_ylim([0.0, None])
            ax_confs.set_xlabel(f'e(s): not {filter_relation}')
            ax_confs.set_ylabel("$\widehat{conf}(R)$")
            ax_confs.set_title(
                "$\widehat{conf}(R)$"
                f" (Avg over {n_random_trials} random selections)\n"
            )
            ax_confs.legend(bbox_to_anchor=(-0.8, 1.0), loc='upper left')
            ax_confs.axvline(x=scar_propensity_score, color='k', linestyle='--')
            ax_confs.text(x=scar_propensity_score, y=conf_scar_y_pos, s="SCAR", verticalalignment='center')

            conf_sqrd_err_max = df_diff_to_true_conf_melted['squared_error'].max()
            conf_sqrd_err_min = df_diff_to_true_conf_melted['squared_error'].min()
            conf_sqrd_err_scar_y_pos = conf_sqrd_err_min + (conf_sqrd_err_max-conf_sqrd_err_min)/2

            ax_squared_error: Axes = sns.lineplot(
                x="prop_score_other",
                y='squared_error',
                hue="error_type",
                # hue_order=order,
                palette=color_palette,
                marker="o",
                data=df_diff_to_true_conf_melted,
                ax=axes[1]
            )
            # ax_confs.legend_ = leg

            ax_squared_error.set_xlim([0.0, 1.0])
            # ax_squared_error.set_ylim([0.0, None])
            ax_squared_error.set_xlabel(f'e(s): not {filter_relation}')
            ax_squared_error.set_ylabel(error_metric_label)
            ax_squared_error.set_title(
                f"{error_metric_label}"
                f" (Avg over {n_random_trials} random selections)\n"
            )
            ax_squared_error.axvline(x=scar_propensity_score, color='k', linestyle='--')
            ax_squared_error.text(x=scar_propensity_score, y=conf_sqrd_err_scar_y_pos, s="SCAR", verticalalignment='center')

            plt.suptitle(f"SAR (2 s-groups: {filter_relation}(s,*)) {rule_name}")
            # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            print(filename_image_confidence_evolution)
            plt.savefig(filename_image_confidence_evolution, bbox_inches='tight')
            plt.close(fig)

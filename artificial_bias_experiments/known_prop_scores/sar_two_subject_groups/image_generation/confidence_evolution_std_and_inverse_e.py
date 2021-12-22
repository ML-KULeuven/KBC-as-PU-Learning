import os
from typing import List, Set, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from artificial_bias_experiments.evaluation.confidence_comparison.df_utils import ColumnNamesInfo, \
    get_df_diffs_between_true_conf_and_confidence_estimators_melted
from artificial_bias_experiments.image_generation.image_generation_utils import \
    get_confidence_difference_label_string

from pylo.language.lp import Clause as PyloClause

from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import get_pylo_rule_from_string, is_pylo_rule_recursive

sns.set(style="whitegrid")


def known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_vs_std_and_inverse_e_per_rule(
        df_rule_wrappers: pd.DataFrame,
        filter_relation: str,
        scar_propensity_score: float,
        image_dir: str,
        filename_root: str,
        separate_recursive_and_non_recursive_rules: bool = True
):
    dir_conf_vs_std_conf_and_inverse_e_conf: str = os.path.join(
        image_dir,
        "confidence_evolution_true_conf_vs_std_and_inverse_e_conf"
    )
    if not os.path.exists(dir_conf_vs_std_conf_and_inverse_e_conf):
        os.makedirs(dir_conf_vs_std_conf_and_inverse_e_conf)

    true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF
    column_names_logistics: List[str] = [
        'target_relation',
        'filter_relation',
        'prop_score_subj',
        'prop_score_other',
        'random_trial_index',
        'Rule',
    ]

    conf_estimators_non_pca_comparison: List[ConfidenceEnum] = [
        ConfidenceEnum.IPW_CONF,
        ConfidenceEnum.CWA_CONF,
        ConfidenceEnum.ICW_CONF
    ]
    confs_non_pca_comparison = [ConfidenceEnum.TRUE_CONF] + conf_estimators_non_pca_comparison

    column_names_non_pca_confidence_comparison = [
        conf.get_name()
        for conf in confs_non_pca_comparison
    ]
    column_names_info = ColumnNamesInfo(
        true_conf=true_conf,
        column_name_true_conf=true_conf.get_name(),
        conf_estimators=conf_estimators_non_pca_comparison,
        column_names_conf_estimators=[
            col.get_name()
            for col in conf_estimators_non_pca_comparison
        ],
        column_names_logistics=column_names_logistics
    )
    color_palette = {
        conf.get_name(): conf.get_hex_color_str()
        for conf in confs_non_pca_comparison
    }

    df_non_pca_conf_comp = df_rule_wrappers[
        column_names_logistics + column_names_non_pca_confidence_comparison
        ]
    n_random_trials = len(df_non_pca_conf_comp["random_trial_index"].unique())

    if separate_recursive_and_non_recursive_rules:
        file_dir_recursive_rules = os.path.join(
            dir_conf_vs_std_conf_and_inverse_e_conf,
            'recursive_rules'
        )
        file_dir_non_recursive_rules = os.path.join(
            dir_conf_vs_std_conf_and_inverse_e_conf,
            'non_recursive_rules'
        )
        if not os.path.exists(file_dir_non_recursive_rules):
            os.makedirs(file_dir_non_recursive_rules)
        if not os.path.exists(file_dir_recursive_rules):
            os.makedirs(file_dir_recursive_rules)

    rule_name: str
    for rule_name in df_rule_wrappers["Rule"].unique():
        df_non_pca_conf_comp_single_rule = df_non_pca_conf_comp[
            df_non_pca_conf_comp["Rule"] == rule_name
            ]

        df_non_pca_conf_comp_melted = df_non_pca_conf_comp_single_rule.melt(
            id_vars=column_names_logistics,
            var_name='confidence_type',
            value_name='confidence_value'
        )

        df_diff_to_true_conf_melted: pd.DataFrame = get_df_diffs_between_true_conf_and_confidence_estimators_melted(
            df_rule_wrappers=df_non_pca_conf_comp_single_rule,
            column_names_info=column_names_info
        )

        file_dir: str
        if separate_recursive_and_non_recursive_rules:
            pylo_rule: PyloClause = get_pylo_rule_from_string(rule_name)
            is_rule_recursive = is_pylo_rule_recursive(pylo_rule)
            if is_rule_recursive:
                file_dir = file_dir_recursive_rules
            else:
                file_dir = file_dir_non_recursive_rules

            filename_image_confidence_evolution: str = os.path.join(
                file_dir,
                f"{filename_root}_conf_estimates_{rule_name}.png"
            )
        else:
            filename_image_confidence_evolution: str = os.path.join(
                dir_conf_vs_std_conf_and_inverse_e_conf,
                f"{filename_root}_conf_estimates_{rule_name}.png"
            )

        figsize = (9, 3)
        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        error_metric_label: str = get_confidence_difference_label_string(true_conf)

        ax_confs: Axes = sns.lineplot(
            x="prop_score_other",
            y='confidence_value',
            hue="confidence_type",
            # hue_order=order,
            palette=color_palette,
            marker="o",
            data=df_non_pca_conf_comp_melted,
            ax=axes[0]
        )
        ax_confs.set_xlim([0.0, 1.0])
        ax_confs.set_ylim([0.0, None])
        ax_confs.set_xlabel(f'e(s): not {filter_relation}')
        # ax_confs.set_ylabel("$\widehat{conf}(R)$")
        ax_confs.set_ylabel("")
        ax_confs.set_title(
            "$\widehat{conf}(R)$",
            loc="left"
            # f" (Avg over {n_random_trials} random selections)\n"
        )

        ax_confs_ymin, ax_confs_ymax = ax_confs.get_ylim()
        # ax_confs_xmin, ax_confs_xmax = ax_confs.get_xlim()
        ax_confs.axvline(x=scar_propensity_score, color='k', linestyle='--')

        ax_confs.text(x=scar_propensity_score,
                      y=(ax_confs_ymax - ax_confs_ymin) / 2 + ax_confs_ymin,
                      s="SCAR",
                      horizontalalignment='right',
                      verticalalignment='center')



        # ax_confs.legend(
        #     bbox_to_anchor=(-0.8, 1.0), loc='upper left'
        # )

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
        ax_squared_error.set_ylim([0.0, None])
        ax_squared_error.set_xlabel(f'e(s): not {filter_relation}')
        # ax_squared_error.set_ylabel(error_metric_label)
        ax_squared_error.set_ylabel("")
        ax_squared_error.set_title(
            f"{error_metric_label}",
            loc="left"
            # f" (Avg over {n_random_trials} random selections)\n"
        )
        ax_squared_error.get_legend().remove()
        ax_squared_error_ymin, ax_squared_error_ymax = ax_squared_error.get_ylim()
        # ax_squared_error_xmin, ax_squared_error_xmax = ax_squared_error.get_xlim()
        ax_squared_error.axvline(x=scar_propensity_score, color='k', linestyle='--')
        ax_squared_error.text(
            x=scar_propensity_score,
            y=(ax_squared_error_ymax - ax_squared_error_ymin) / 2 + ax_squared_error_ymin,
            s="SCAR",
            horizontalalignment='right',
            verticalalignment='center'
        )


        handles, labels = ax_confs.get_legend_handles_labels()
        from matplotlib.legend import Legend
        ax_confs_tmp_legend: Legend = ax_confs.get_legend()
        legend_rectangle = ax_confs_tmp_legend.get_frame()
        legend_width = legend_rectangle.get_width()

        ax_confs.legend(handles=handles[1:],
                        labels=labels[1:],
                        bbox_to_anchor=(-0.2, 0.5),
                        loc='center right',
                        title='confidence')

        plt.suptitle(f"{rule_name}\nq={filter_relation}")
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
        plt.savefig(filename_image_confidence_evolution, bbox_inches='tight')
        plt.close(fig)

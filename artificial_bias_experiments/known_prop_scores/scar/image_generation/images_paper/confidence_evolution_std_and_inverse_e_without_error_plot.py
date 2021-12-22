import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pylo.language.lp import Clause as PyloClause

from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import get_pylo_rule_from_string, is_pylo_rule_recursive

sns.set(style="whitegrid")


def known_prop_scores_scar_plot_conf_evolution_true_conf_vs_std_and_inverse_e_per_rule_without_error(
        df_rule_wrappers: pd.DataFrame,
        image_dir: str,
        filename_root: str,
        separate_recursive_and_non_recursive_rules: bool = True
):
    dir_conf_vs_std_conf_and_inverse_e_conf: str = os.path.join(
        image_dir,
        "confidence_evolution_true_conf_vs_std_and_inverse_e_conf_without_error"
    )
    if not os.path.exists(dir_conf_vs_std_conf_and_inverse_e_conf):
        os.makedirs(dir_conf_vs_std_conf_and_inverse_e_conf)

    column_names_logistics: List[str] = [
        'target_relation',
        'label_frequency',
        'random_trial_index',
        "Rule"
    ]

    true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF

    conf_estimators_non_pca_comparison: List[ConfidenceEnum] = [
        ConfidenceEnum.IPW_CONF,
        ConfidenceEnum.CWA_CONF,
    ]

    confs_non_pca_comparison = [true_conf] + conf_estimators_non_pca_comparison

    column_names_non_pca_confidence_comparison = [
        conf.get_name()
        for conf in confs_non_pca_comparison
    ]

    color_palette = {
        conf.get_name(): conf.get_hex_color_str()
        for conf in confs_non_pca_comparison
    }

    df_non_pca_conf_comp = df_rule_wrappers[
        column_names_logistics + column_names_non_pca_confidence_comparison
        ]

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

    list_of_rules: List[str] = list(df_rule_wrappers["Rule"].unique())
    rule_name: str
    for rule_name in list_of_rules:
        # True Conf vs non-pca estimators
        df_non_pca_conf_comp_single_rule = df_non_pca_conf_comp[
            df_non_pca_conf_comp["Rule"] == rule_name
            ]

        df_non_pca_conf_comp_melted = df_non_pca_conf_comp_single_rule.melt(
            id_vars=column_names_logistics,
            var_name='confidence_type',
            value_name='confidence_value'
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

        # Plot True conf, conf_STD and conf_{1/e}
        figsize = (6, 3)
        fig, axes = plt.subplots(1, 1, figsize=figsize)

        fig: Figure
        axes: Axes
        ax_confs: Axes = sns.lineplot(
            x="label_frequency",
            y='confidence_value',
            hue="confidence_type",
            palette=color_palette,
            marker="o",
            data=df_non_pca_conf_comp_melted,
            ax=axes
        )
        ax_confs.set_xlim([0.0, 1.0])
        ax_confs.set_ylim([0.0, None])
        ax_confs.set_xlabel(f'label frequency c')
        ax_confs.set_ylabel("")
        ax_confs.set_title("$\widehat{conf}(R)$",
                           loc="left"
                           )

        # handles, labels = ax_confs.get_legend_handles_labels()
        # from matplotlib.legend import Legend
        # ax_confs_tmp_legend: Legend = ax_confs.get_legend()
        # legend_rectangle = ax_confs_tmp_legend.get_frame()
        # legend_width = legend_rectangle.get_width()

        ax_confs.legend(
            # handles=handles[1:],
            # labels=labels[1:],
            bbox_to_anchor=(-0.1, 0.5),
            loc='center right',
            title='confidence')

        plt.suptitle(f"{rule_name}", y=1.03)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        plt.tight_layout()
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename_image_confidence_evolution, bbox_inches='tight')
        plt.close(fig)

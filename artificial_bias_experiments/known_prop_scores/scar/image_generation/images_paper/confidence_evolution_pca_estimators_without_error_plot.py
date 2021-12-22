import os
from typing import List, Dict, Tuple

import pandas as pd
from matplotlib.lines import Line2D

from pylo.language.lp import Clause as PyloClause

from artificial_bias_experiments.images_paper_joint.pretty_rule_string import \
    get_paper_like_rule_string_from_prolog_str
from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import get_pylo_rule_from_string, is_pylo_rule_recursive

import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.axes import Axes
from matplotlib.figure import Figure

sns.set(style="whitegrid")


def get_df_pca_confs_scar(
        df_rule_wrappers: pd.DataFrame,
        column_names_logistics: List[str],
        indicator_string: str

) -> Tuple[pd.DataFrame, Dict]:

    if indicator_string == "PCA":
        conf_estimators_pca_conf_comparison: List[ConfidenceEnum] = [
            ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O,
            ConfidenceEnum.PCA_CONF_S_TO_O,

            ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S,
            ConfidenceEnum.PCA_CONF_O_TO_S
        ]
    elif indicator_string == "IPW-PCA":
        conf_estimators_pca_conf_comparison: List[ConfidenceEnum] = [
            ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O,
            ConfidenceEnum.IPW_PCA_CONF_S_TO_O,

            ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S,
            ConfidenceEnum.IPW_PCA_CONF_O_TO_S
        ]
    else:
        raise Exception("Unexpected indicator string")

    true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF
    confs_pca_comparison = [true_conf] + conf_estimators_pca_conf_comparison
    column_names_pca_confidence_comparison = [
        conf.get_name()
        for conf in confs_pca_comparison
    ]

    color_palette = {
        conf.get_name(): conf.get_hex_color_str()
        for conf in confs_pca_comparison
    }

    df_pca_conf_comp = df_rule_wrappers[
        column_names_logistics + column_names_pca_confidence_comparison
        ]
    return df_pca_conf_comp, color_palette


def plot_conf_evolution_pca_like_estimators_scar(
        df_pca_conf_comp_single_rule_melted: pd.DataFrame,
        color_palette: Dict,
        axis: Axes
) -> Axes:
    ax_confs_pca: Axes = sns.lineplot(
        x="label_frequency",
        y='confidence_value',
        hue="confidence_type",
        palette=color_palette,
        # marker="o",
        data=df_pca_conf_comp_single_rule_melted,
        ax=axis
    )
    ax_confs_pca.set_xlim([0.0, 1.0])
    ax_confs_pca.set_ylim([0.0, None])
    ax_confs_pca.set_xlabel('$c_{p}$')

    ax_confs_pca.set_ylabel("$\widehat{conf}(R)$")
    # ax_confs_pca.set_title("$\widehat{conf}(R)$",
    #                        loc="left"
    #                        )
    ax_confs_pca.grid(False)
    return ax_confs_pca


def known_prop_scores_scar_plot_conf_evolution_true_conf_vs_pca_estimators_per_rule_without_error(
    df_rule_wrappers: pd.DataFrame,
    image_dir: str,
    filename_root: str,
    separate_recursive_and_non_recursive_rules: bool = True
):
    dir_conf_vs_pca_confs: str = os.path.join(
        image_dir,
        "confidence_evolution_true_conf_vs_pca_confidences_without_error"
    )
    if not os.path.exists(dir_conf_vs_pca_confs):
        os.makedirs(dir_conf_vs_pca_confs)

    column_names_logistics: List[str] = [
        'target_relation',
        'label_frequency',
        'random_trial_index',
        "Rule"
    ]

    df_pca_conf_comp, color_palette_pca_confs = get_df_pca_confs_scar(
        df_rule_wrappers=df_rule_wrappers,
        column_names_logistics=column_names_logistics,
        indicator_string="PCA"
    )
    df_ipw_pca_conf_comp, color_palette_ipw_pca_confs = get_df_pca_confs_scar(
        df_rule_wrappers=df_rule_wrappers,
        column_names_logistics=column_names_logistics,
        indicator_string="IPW-PCA"
    )
    color_palette: Dict[str, str] = {**color_palette_pca_confs, **color_palette_ipw_pca_confs}

    if separate_recursive_and_non_recursive_rules:
        file_dir_recursive_rules = os.path.join(
            dir_conf_vs_pca_confs,
            'recursive_rules'
        )
        file_dir_non_recursive_rules = os.path.join(
            dir_conf_vs_pca_confs,
            'non_recursive_rules'
        )
        if not os.path.exists(file_dir_non_recursive_rules):
            os.makedirs(file_dir_non_recursive_rules)
        if not os.path.exists(file_dir_recursive_rules):
            os.makedirs(file_dir_recursive_rules)

    list_of_rules: List[str] = list(df_rule_wrappers["Rule"].unique())
    rule_name: str
    for rule_name in list_of_rules:
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
                f"{filename_root}_pca_conf_estimates_{rule_name}.png"
            )
        else:
            filename_image_confidence_evolution: str = os.path.join(
                dir_conf_vs_pca_confs,
                f"{filename_root}_pca_conf_estimates_{rule_name}.png"
            )

        # True Conf vs non-pca estimators
        df_pca_conf_comp_single_rule = df_pca_conf_comp[
            df_pca_conf_comp["Rule"] == rule_name
        ]
        df_pca_conf_comp_single_rule_melted = df_pca_conf_comp_single_rule.melt(
            id_vars=column_names_logistics,
            var_name='confidence_type',
            value_name='confidence_value'
        )

        df_ipw_pca_conf_comp_single_rule = df_ipw_pca_conf_comp[
            df_ipw_pca_conf_comp["Rule"] == rule_name
            ]
        df_ipw_pca_conf_comp_single_rule_melted = df_ipw_pca_conf_comp_single_rule.melt(
            id_vars=column_names_logistics,
            var_name='confidence_type',
            value_name='confidence_value'
        )

        figsize = (9, 3)

        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax_pca_confs: Axes = plot_conf_evolution_pca_like_estimators_scar(
            df_pca_conf_comp_single_rule_melted=df_pca_conf_comp_single_rule_melted,
            color_palette=color_palette,
            axis=axes[0]
        )
        ax_ipw_pca_confs: Axes = plot_conf_evolution_pca_like_estimators_scar(
            df_pca_conf_comp_single_rule_melted=df_ipw_pca_conf_comp_single_rule_melted,
            color_palette=color_palette,
            axis=axes[1]
        )

        ax_pca_confs_handles: List[Line2D]
        ax_pca_confs_labels: List[str]
        ax_ipw_pca_confs_handles: List[Line2D]
        ax_ipw_pca_confs_labels: List[str]

        ax_pca_confs_handles, ax_pca_confs_labels = ax_pca_confs.get_legend_handles_labels()
        ax_ipw_pca_confs_handles, ax_ipw_pca_confs_labels = ax_ipw_pca_confs.get_legend_handles_labels()
        for handle, label in zip(ax_ipw_pca_confs_handles, ax_ipw_pca_confs_labels):
            if label not in ax_pca_confs_labels:
                ax_pca_confs_handles.append(handle)
                ax_pca_confs_labels.append(label)

        sorted_labels_and_handles: List[Tuple[str, Line2D]] = [
            (label, handle)
            for label, handle in sorted(zip(ax_pca_confs_labels, ax_pca_confs_handles))
        ]
        unzipped_sorted_labels_and_handles = list(zip(*sorted_labels_and_handles))
        ax_pca_confs_labels = list(unzipped_sorted_labels_and_handles[0])
        ax_pca_confs_handles = list(unzipped_sorted_labels_and_handles[1])

        ax_pca_confs.legend(
            handles=ax_pca_confs_handles,
            labels=ax_pca_confs_labels,
            bbox_to_anchor=(-0.15, 0.5),
            loc='center right',
            title="confidence"
        )
        ax_ipw_pca_confs.get_legend().remove()

        paper_like_rule_string: str = get_paper_like_rule_string_from_prolog_str(rule_name)
        plt.suptitle(paper_like_rule_string, y=1.03)

        plt.tight_layout()  # rect=[left, bottom, right, top]
        # plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
        plt.savefig(filename_image_confidence_evolution, bbox_inches='tight')
        plt.close(fig)

import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pylo.language.lp import Clause as PyloClause

from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import get_pylo_rule_from_string, is_pylo_rule_recursive

sns.set(style="whitegrid")

def get_df_pca_confs_sar_two_groups(
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


def plot_conf_evolution_pca_like_estimators_sar_two_groups(
        df_pca_conf_comp_single_rule_melted: pd.DataFrame,
        scar_propensity_score: float,
        filter_relation: str,
        color_palette: Dict,
        axis: Axes
) -> Axes:
    ax_confs: Axes = sns.lineplot(
        x="prop_score_other",
        y='confidence_value',
        hue="confidence_type",
        # hue_order=order,
        palette=color_palette,
        # marker="o",
        data=df_pca_conf_comp_single_rule_melted,
        ax=axis
    )
    ax_confs.set_xlim([0.0, 1.0])
    ax_confs.set_ylim([0.0, None])
    # ax_confs.set_xlabel("$c_{\\neg q}$")
    ax_confs.set_xlabel("$c_{\\neg q}$, q=" + filter_relation)
    ax_confs.set_ylabel("")
    # ax_confs.set_title(
    #     "$\widehat{conf}(R)$",
    #     loc="left"
    # )
    ax_confs.grid(False)

    # ax_confs_ymin, ax_confs_ymax = ax_confs.get_ylim()
    # ax_confs.axvline(x=scar_propensity_score, color='k', linestyle='--')
    # ax_confs.text(x=scar_propensity_score,
    #               y=(ax_confs_ymax - ax_confs_ymin) / 2 + ax_confs_ymin,
    #               s="SCAR",
    #               horizontalalignment='right',
    #               verticalalignment='center')
    return ax_confs


def known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule_without_error(
        df_rule_wrappers: pd.DataFrame,
        filter_relation: str,
        scar_propensity_score: float,
        image_dir: str,
        filename_root: str,
        separate_recursive_and_non_recursive_rules: bool = True
):
    dir_true_conf_vs_pca_confidence_estimators: str = os.path.join(
        image_dir,
        "confidence_evolution_true_conf_star_vs_pca_estimators_without_error"
    )
    if not os.path.exists(dir_true_conf_vs_pca_confidence_estimators):
        os.makedirs(dir_true_conf_vs_pca_confidence_estimators)

    column_names_logistics: List[str] = [
        'target_relation',
        'filter_relation',
        'prop_score_subj',
        'prop_score_other',
        'random_trial_index',
        'Rule',
    ]

    df_pca_conf_comp, color_palette_pca_confs = get_df_pca_confs_sar_two_groups(
        df_rule_wrappers=df_rule_wrappers,
        column_names_logistics=column_names_logistics,
        indicator_string="PCA"
    )
    df_ipw_pca_conf_comp, color_palette_ipw_pca_confs = get_df_pca_confs_sar_two_groups(
        df_rule_wrappers=df_rule_wrappers,
        column_names_logistics=column_names_logistics,
        indicator_string="IPW-PCA"
    )
    color_palette: Dict[str, str] = {**color_palette_pca_confs, **color_palette_ipw_pca_confs}

    if separate_recursive_and_non_recursive_rules:
        file_dir_recursive_rules = os.path.join(
            dir_true_conf_vs_pca_confidence_estimators,
            'recursive_rules'
        )
        file_dir_non_recursive_rules = os.path.join(
            dir_true_conf_vs_pca_confidence_estimators,
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
                dir_true_conf_vs_pca_confidence_estimators,
                f"{filename_root}_pca_conf_estimates_{rule_name}.png"
            )

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

        ax_pca_confs: Axes = plot_conf_evolution_pca_like_estimators_sar_two_groups(
            df_pca_conf_comp_single_rule_melted=df_pca_conf_comp_single_rule_melted,
            scar_propensity_score=scar_propensity_score,
            filter_relation=filter_relation,
            color_palette=color_palette,
            axis=axes[0]
        )
        ax_ipw_pca_confs: Axes = plot_conf_evolution_pca_like_estimators_sar_two_groups(
            df_pca_conf_comp_single_rule_melted=df_ipw_pca_conf_comp_single_rule_melted,
            scar_propensity_score=scar_propensity_score,
            filter_relation=filter_relation,
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

        plt.suptitle(f"{rule_name}", y=1.03)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        plt.tight_layout()  # rect=[left, bottom, right, top]
        # plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
        plt.savefig(filename_image_confidence_evolution, bbox_inches='tight')
        plt.close(fig)

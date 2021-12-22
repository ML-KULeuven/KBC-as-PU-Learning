import os
from typing import List, Set, Dict

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from artificial_bias_experiments.images_paper_joint.pretty_rule_string import \
    get_paper_like_rule_string_from_prolog_str

from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import get_pylo_rule_from_string, is_pylo_rule_recursive

from pylo.language.lp import Clause as PyloClause

sns.set_style(style="white")

column_names_logistics_sar_two_groups: List[str] = [
    'target_relation',
    'filter_relation',
    'prop_score_subj',
    'prop_score_other',
    'random_trial_index',
    'Rule',
]

column_names_logistics_scar: List[str] = [
    'target_relation',
    'label_frequency',
    'random_trial_index',
    "Rule"
]


def plot_cwa_scar(
        df_non_pca_conf_comp_melted: pd.DataFrame,
        axis: Axes,
        color_palette: Dict[str, str],
) -> Axes:
    ax_confs: Axes = sns.lineplot(
        x="label_frequency",
        y='confidence_value',
        hue="confidence_type",
        palette=color_palette,
        # marker="o",
        data=df_non_pca_conf_comp_melted,
        ax=axis
    )

    ax_confs.set_xlim([0.0, 1.0])
    ax_confs.set_ylim([0.0, None])
    ax_confs.set_xlabel('$c_{p}$')
    ax_confs.set_ylabel("$\widehat{conf}(R)$",)
    ax_confs.set_title(
        "${SCAR}_{p}$",
        # "$\widehat{conf}(R)$",
        # loc="left"
    )
    ax_confs.grid(False)
    ax_confs.set_xticks([0.0, 1.0])

    return ax_confs


def plot_cwa_sar_two_groups(
        df_non_pca_conf_comp_melted: pd.DataFrame,
        scar_propensity_score: float,
        filter_relation: str,
        axis: Axes,
        color_palette: Dict[str, str],

) -> Axes:
    ax_confs: Axes = sns.lineplot(
        x="prop_score_other",
        y='confidence_value',
        hue="confidence_type",
        # hue_order=order,
        palette=color_palette,
        # marker="o",
        data=df_non_pca_conf_comp_melted,
        ax=axis
    )
    ax_confs.set_xlim([0.0, 1.0])
    ax_confs.set_ylim([0.0, None])
    ax_confs.set_xticks([0.0, 1.0])
    ax_confs.set_xlabel("$c_{\\neg q}$, q=" + filter_relation)
    # ax_confs.set_ylabel("$\widehat{conf}(R)$")
    ax_confs.set_ylabel("")
    ax_confs.set_title(
        # "$\widehat{conf}(R)$",
        "${SAR}_{group}$",
        # loc="left"
        # f" (Avg over {n_random_trials} random selections)\n"
    )
    ax_confs.grid(False)

    # ax_confs_ymin, ax_confs_ymax = ax_confs.get_ylim()
    # ax_confs_xmin, ax_confs_xmax = ax_confs.get_xlim()
    # ax_confs.axvline(x=scar_propensity_score, color='k', linestyle='--')

    # ax_confs.text(x=scar_propensity_score,
    #               y=(ax_confs_ymax - ax_confs_ymin) / 2 + ax_confs_ymin,
    #               s="SCAR",
    #               horizontalalignment='right',
    #               verticalalignment='center')
    return ax_confs


def prepare_scar(df_rule_wrappers: pd.DataFrame):
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
        column_names_logistics_scar + column_names_non_pca_confidence_comparison
        ]
    return df_non_pca_conf_comp, color_palette


def prepare_sar_two_groups(df_rule_wrappers: pd.DataFrame):
    conf_estimators_non_pca_comparison: List[ConfidenceEnum] = [
        ConfidenceEnum.IPW_CONF,
        ConfidenceEnum.CWA_CONF,
        ConfidenceEnum.ICW_CONF
    ]
    true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF
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
        column_names_logistics_sar_two_groups + column_names_non_pca_confidence_comparison
        ]
    return df_non_pca_conf_comp, color_palette


def prep_file_dir_recursive_and_non_recursive(dir_cwa_conf_evol: str):
    file_dir_recursive_rules = os.path.join(
        dir_cwa_conf_evol,
        'recursive_rules'
    )
    file_dir_non_recursive_rules = os.path.join(
        dir_cwa_conf_evol,
        'non_recursive_rules'
    )
    if not os.path.exists(file_dir_non_recursive_rules):
        os.makedirs(file_dir_non_recursive_rules)
    if not os.path.exists(file_dir_recursive_rules):
        os.makedirs(file_dir_recursive_rules)

    return file_dir_recursive_rules, file_dir_non_recursive_rules


def generate_cwa_conf_image_scar_and_sar_two_groups(
        df_rule_wrappers_scar: pd.DataFrame,
        df_rule_wrappers_sar_two_groups: pd.DataFrame,
        target_relation: str,
        filter_relation_sar_two_groups: str,
        scar_propensity_score_sar_two_groups: float,
        image_dir: str
) -> None:
    set_of_scar_rules: Set[str] = set(df_rule_wrappers_scar["Rule"].unique())
    set_of_sar_two_groups_rules: Set[str] = set(df_rule_wrappers_sar_two_groups["Rule"].unique())

    intersection_of_rules_set: Set[str] = set_of_sar_two_groups_rules.intersection(set_of_scar_rules)
    intersection_of_rules_list: List[str] = list(sorted(intersection_of_rules_set))

    if len(intersection_of_rules_list) > 0:
        dir_cwa_conf_evol: str = os.path.join(
            image_dir,
            "cwa_conf_evolution"
        )
        if not os.path.exists(dir_cwa_conf_evol):
            os.makedirs(dir_cwa_conf_evol)
        print(dir_cwa_conf_evol)

        file_dir_recursive_rules: str
        file_dir_non_recursive_rules: str
        file_dir_recursive_rules, file_dir_non_recursive_rules = prep_file_dir_recursive_and_non_recursive(
            dir_cwa_conf_evol=dir_cwa_conf_evol
        )

        df_non_pca_conf_comp_scar: pd.DataFrame
        color_palette_scar: Dict[str, str]
        df_non_pca_conf_comp_scar, color_palette_scar = prepare_scar(
            df_rule_wrappers=df_rule_wrappers_scar
        )

        df_non_pca_conf_comp_sar_two_groups: pd.DataFrame
        color_palette_sar_two_groups: Dict[str, str]
        df_non_pca_conf_comp_sar_two_groups, color_palette_sar_two_groups = prepare_sar_two_groups(
            df_rule_wrappers=df_rule_wrappers_sar_two_groups
        )
        color_palette: Dict[str, str] = {**color_palette_scar, **color_palette_sar_two_groups}

        for rule_name in intersection_of_rules_list:
            file_dir: str
            pylo_rule: PyloClause = get_pylo_rule_from_string(rule_name)
            is_rule_recursive = is_pylo_rule_recursive(pylo_rule)
            if is_rule_recursive:
                file_dir = file_dir_recursive_rules
            else:
                file_dir = file_dir_non_recursive_rules

            filename_image_confidence_evolution: str = os.path.join(
                file_dir,
                f"cwa_evol_{target_relation}_{filter_relation_sar_two_groups}_{rule_name}.png"
            )

            df_non_pca_conf_comp_scar_single_rule = df_non_pca_conf_comp_scar[
                df_non_pca_conf_comp_scar["Rule"] == rule_name
                ]

            df_non_pca_conf_comp_scar_single_rule_melted = df_non_pca_conf_comp_scar_single_rule.melt(
                id_vars=column_names_logistics_scar,
                var_name='confidence_type',
                value_name='confidence_value'
            )

            df_non_pca_conf_comp_sar_two_groups_single_rule = df_non_pca_conf_comp_sar_two_groups[
                df_non_pca_conf_comp_sar_two_groups["Rule"] == rule_name
                ]
            df_non_pca_conf_comp_sar_two_groups_single_rule_melted = df_non_pca_conf_comp_sar_two_groups_single_rule.melt(
                id_vars=column_names_logistics_sar_two_groups,
                var_name='confidence_type',
                value_name='confidence_value'
            )

            figsize = (6, 2.5)
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

            ax_scar: Axes = plot_cwa_scar(
                df_non_pca_conf_comp_melted=df_non_pca_conf_comp_scar_single_rule_melted,
                axis=axes[0],
                color_palette=color_palette
            )
            ax_scar.get_legend().remove()

            ax_sar_two_groups: Axes = plot_cwa_sar_two_groups(
                df_non_pca_conf_comp_melted=df_non_pca_conf_comp_sar_two_groups_single_rule_melted,
                scar_propensity_score=scar_propensity_score_sar_two_groups,
                filter_relation=filter_relation_sar_two_groups,
                axis=axes[1],
                color_palette=color_palette
            )
            ax_sar_two_groups.grid(False)
            ax_sar_two_groups.legend(
                # handles=handles[1:],
                # labels=labels[1:],
                bbox_to_anchor=(1.02, 0.5),
                loc='center left',
                title='confidence')

            sns.despine()

            paper_like_rule_string: str = get_paper_like_rule_string_from_prolog_str(rule_name)
            plt.suptitle(paper_like_rule_string, y=1.03)
            plt.tight_layout()  # rect=[left, bottom, right, top]
            # plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
            plt.savefig(filename_image_confidence_evolution, bbox_inches='tight')
            plt.close(fig)

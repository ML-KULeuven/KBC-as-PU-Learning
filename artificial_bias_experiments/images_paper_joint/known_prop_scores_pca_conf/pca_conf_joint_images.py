import os

import pandas as pd
from typing import List, Set, Dict, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from artificial_bias_experiments.images_paper_joint.known_prop_scores_cwa_conf.cwa_conf_joint_images import \
    prep_file_dir_recursive_and_non_recursive
from artificial_bias_experiments.images_paper_joint.pretty_rule_string import \
    get_paper_like_rule_string_from_prolog_str
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.images_paper.confidence_evolution_pca_estimators_without_error import \
    get_df_pca_confs_sar_two_groups, plot_conf_evolution_pca_like_estimators_sar_two_groups
from artificial_bias_experiments.known_prop_scores.scar.image_generation.images_paper.confidence_evolution_pca_estimators_without_error_plot import \
    get_df_pca_confs_scar, plot_conf_evolution_pca_like_estimators_scar
from kbc_pul.confidence_naming import ConfidenceEnum

from pylo.language.lp import Clause as PyloClause

from kbc_pul.data_structures.rule_wrapper import is_pylo_rule_recursive, get_pylo_rule_from_string

column_names_logistics_scar: List[str] = [
        'target_relation',
        'label_frequency',
        'random_trial_index',
        "Rule"
    ]

column_names_logistics_sar_two_groups: List[str] = [
    'target_relation',
    'filter_relation',
    'prop_score_subj',
    'prop_score_other',
    'random_trial_index',
    'Rule',
]


def generate_pca_conf_image_scar_and_sar_two_groups(
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
            "pca_conf_evolution"
        )
        if not os.path.exists(dir_cwa_conf_evol):
            os.makedirs(dir_cwa_conf_evol)
        print(dir_cwa_conf_evol)

        file_dir_recursive_rules: str
        file_dir_non_recursive_rules: str
        file_dir_recursive_rules, file_dir_non_recursive_rules = prep_file_dir_recursive_and_non_recursive(
            dir_cwa_conf_evol=dir_cwa_conf_evol
        )

        # SCAR ############################
        df_pca_conf_comp_scar: pd.DataFrame
        color_palette_pca_confs_scar: Dict[str, str]
        df_pca_conf_comp_scar, color_palette_pca_confs_scar = get_df_pca_confs_scar(
            df_rule_wrappers=df_rule_wrappers_scar,
            column_names_logistics=column_names_logistics_scar,
            indicator_string="PCA"
        )

        df_ipw_pca_conf_comp_scar: pd.DataFrame
        color_palette_ipw_pca_confs: Dict[str, str]
        df_ipw_pca_conf_comp_scar, color_palette_ipw_pca_confs_scar = get_df_pca_confs_scar(
            df_rule_wrappers=df_rule_wrappers_scar,
            column_names_logistics=column_names_logistics_scar,
            indicator_string="IPW-PCA"
        )
        # SAR TWO GROUPS #############################################
        df_pca_conf_comp_sar_two_groups, color_palette_pca_confs_sar_two_groups = get_df_pca_confs_sar_two_groups(
            df_rule_wrappers=df_rule_wrappers_sar_two_groups,
            column_names_logistics=column_names_logistics_sar_two_groups,
            indicator_string="PCA"
        )
        df_ipw_pca_conf_comp_sar_two_groups, color_palette_ipw_pca_confs_sar_two_groups = get_df_pca_confs_sar_two_groups(
            df_rule_wrappers=df_rule_wrappers_sar_two_groups,
            column_names_logistics=column_names_logistics_sar_two_groups,
            indicator_string="IPW-PCA"
        )

        color_palette: Dict[str, str] = {
            **color_palette_pca_confs_scar, **color_palette_ipw_pca_confs_scar,
            **color_palette_pca_confs_sar_two_groups, **color_palette_ipw_pca_confs_sar_two_groups

        }
        
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
                f"pca_evol_{target_relation}_{filter_relation_sar_two_groups}_{rule_name}.png"
            )

            # SCAR ############################
            df_pca_conf_comp_scar_single_rule = df_pca_conf_comp_scar[
                df_pca_conf_comp_scar["Rule"] == rule_name
                ]
            df_pca_conf_comp_scar_single_rule_melted = df_pca_conf_comp_scar_single_rule.melt(
                id_vars=column_names_logistics_scar,
                var_name='confidence_type',
                value_name='confidence_value'
            )

            df_ipw_pca_conf_comp_scar_single_rule = df_ipw_pca_conf_comp_scar[
                df_ipw_pca_conf_comp_scar["Rule"] == rule_name
                ]
            df_ipw_pca_conf_comp_scar_single_rule_melted = df_ipw_pca_conf_comp_scar_single_rule.melt(
                id_vars=column_names_logistics_scar,
                var_name='confidence_type',
                value_name='confidence_value'
            )

            # SAR TWO GROUPS #############################################
            df_pca_conf_comp_sar_two_groups_single_rule = df_pca_conf_comp_sar_two_groups[
                df_pca_conf_comp_sar_two_groups["Rule"] == rule_name
                ]
            df_pca_conf_comp_sar_two_groups_single_rule_melted = df_pca_conf_comp_sar_two_groups_single_rule.melt(
                id_vars=column_names_logistics_sar_two_groups,
                var_name='confidence_type',
                value_name='confidence_value'
            )

            df_ipw_pca_conf_comp_sar_two_groups_single_rule = df_ipw_pca_conf_comp_sar_two_groups[
                df_ipw_pca_conf_comp_sar_two_groups["Rule"] == rule_name
                ]
            df_ipw_pca_conf_comp_sar_two_groups_single_rule_melted = df_ipw_pca_conf_comp_sar_two_groups_single_rule.melt(
                id_vars=column_names_logistics_sar_two_groups,
                var_name='confidence_type',
                value_name='confidence_value'
            )

            fig_width = 6
            fig_height = 3
            figsize = (fig_width, fig_height)
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)


            # SCAR ##################################
            ax_pca_confs_scar: Axes = plot_conf_evolution_pca_like_estimators_scar(
                df_pca_conf_comp_single_rule_melted=df_pca_conf_comp_scar_single_rule_melted,
                color_palette=color_palette,
                axis=axes[0, 0]
            )
            # ax_pca_confs_scar.set_xlabel("")
            # ax_pca_confs_scar.set_xticks([])
            ax_pca_confs_scar.set_title(
                # "$\widehat{conf}(R)$",
                "${SCAR}_{p}$",
                # loc="left"
                # f" (Avg over {n_random_trials} random selections)\n"
            )


            ax_ipw_pca_confs_scar: Axes = plot_conf_evolution_pca_like_estimators_scar(
                df_pca_conf_comp_single_rule_melted=df_ipw_pca_conf_comp_scar_single_rule_melted,
                color_palette=color_palette,
                axis=axes[1,0]
            )
            ax_ipw_pca_confs_scar.set_xticks([0.0, 1.0])

            ax_pca_confs_sar_two_groups: Axes = plot_conf_evolution_pca_like_estimators_sar_two_groups(
                df_pca_conf_comp_single_rule_melted=df_pca_conf_comp_sar_two_groups_single_rule_melted,
                scar_propensity_score=scar_propensity_score_sar_two_groups,
                filter_relation=filter_relation_sar_two_groups,
                color_palette=color_palette,
                axis=axes[0,1]
            )
            # ax_pca_confs_sar_two_groups.set_xlabel("")
            # ax_pca_confs_sar_two_groups.set_xticks([])
            ax_pca_confs_sar_two_groups.set_title(
                # "$\widehat{conf}(R)$",
                "${SAR}_{group}$",
                # loc="left"
                # f" (Avg over {n_random_trials} random selections)\n"
            )


            ax_ipw_pca_confs_sar_two_groups: Axes = plot_conf_evolution_pca_like_estimators_sar_two_groups(
                df_pca_conf_comp_single_rule_melted=df_ipw_pca_conf_comp_sar_two_groups_single_rule_melted,
                scar_propensity_score=scar_propensity_score_sar_two_groups,
                filter_relation=filter_relation_sar_two_groups,
                color_palette=color_palette,
                axis=axes[1,1]
            )
            ax_ipw_pca_confs_sar_two_groups.set_xticks([0.0, 1.0])

            ax_pca_confs_handles: List[Line2D]
            ax_pca_confs_labels: List[str]
            ax_ipw_pca_confs_handles: List[Line2D]
            ax_ipw_pca_confs_labels: List[str]


            label_order = [
                ConfidenceEnum.TRUE_CONF.value,

                ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O.value,
                ConfidenceEnum.PCA_CONF_S_TO_O.value,
                ConfidenceEnum.IPW_PCA_CONF_S_TO_O.value,


                ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S.value,
                ConfidenceEnum.PCA_CONF_O_TO_S.value,
                ConfidenceEnum.IPW_PCA_CONF_O_TO_S.value,

            ]

            ax_pca_confs_handles, ax_pca_confs_labels = ax_pca_confs_sar_two_groups.get_legend_handles_labels()
            ax_ipw_pca_confs_handles, ax_ipw_pca_confs_labels = ax_ipw_pca_confs_sar_two_groups.get_legend_handles_labels()
            for handle, label in zip(ax_ipw_pca_confs_handles, ax_ipw_pca_confs_labels):
                if label not in ax_pca_confs_labels:
                    ax_pca_confs_handles.append(handle)
                    ax_pca_confs_labels.append(label)


            sorted_labels_and_handles: List[Tuple[str, Line2D]] = [
                (label, handle)
                for label, handle in sorted(zip(ax_pca_confs_labels, ax_pca_confs_handles))
            ]

            tmp_handles = []
            for label_tmp in label_order:
                for label_tmp2, handle_tmp2 in sorted_labels_and_handles:
                    if label_tmp == label_tmp2:
                        tmp_handles.append(handle_tmp2)


            unzipped_sorted_labels_and_handles = list(zip(*sorted_labels_and_handles))
            # ax_pca_confs_labels = list(unzipped_sorted_labels_and_handles[0])
            # ax_pca_confs_handles = list(unzipped_sorted_labels_and_handles[1])
            ax_pca_confs_labels = label_order
            ax_pca_confs_handles = tmp_handles

            ax_pca_confs_scar.get_legend().remove()
            ax_ipw_pca_confs_scar.get_legend().remove()
            ax_pca_confs_sar_two_groups.get_legend().remove()
            ax_ipw_pca_confs_sar_two_groups.get_legend().remove()

            # ax_ipw_pca_confs_sar_two_groups.legend(
            #     handles=ax_pca_confs_handles,
            #     labels=ax_pca_confs_labels,
            #     bbox_to_anchor=(1.01, 0.5),
            #     loc='center left',
            #     title="confidence"
            # )
            fig.legend(
                handles=ax_pca_confs_handles,
                labels=ax_pca_confs_labels,
                title="confidence",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                # ncol=4
            )

            sns.despine()
            paper_like_rule_string: str = get_paper_like_rule_string_from_prolog_str(rule_name)
            plt.suptitle(paper_like_rule_string, y=1.03)

            plt.tight_layout()  # rect=[left, bottom, right, top]
            # plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
            plt.savefig(filename_image_confidence_evolution, bbox_inches='tight')
            plt.close(fig)


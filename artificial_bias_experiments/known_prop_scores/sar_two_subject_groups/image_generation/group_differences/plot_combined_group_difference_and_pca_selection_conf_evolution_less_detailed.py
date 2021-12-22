import os
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from pylo.language.lp import Clause as PyloClause

from artificial_bias_experiments.images_paper_joint.pretty_rule_string import \
    get_paper_like_rule_string_from_prolog_str
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.bar_plot_utils import \
    show_values_on_bars
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.column_names import \
    GroupNameEnum, CNameEnum
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.loading_df_with_group_info_using_known_prop_scores_sar import \
    get_dataframe_with_info_about_known_prop_scores_sar_two_groups__for
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.melting_dataframes import (
    MeltedDataFrameInfo,
    get_df_abs_conf_per_group, get_df_rel_conf_per_group, get_df_abs_n_predictions_per_group,
    get_df_rel_n_predictions_per_group, get_df_abs_pair_pos_conf_s_to_o_per_group
)
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.pca_selection_confidence_evolution_v2 import \
    plot_known_prop_score_pca_selection_for_true_conf_star_on_axis
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.known_prop_scores_sar_generate_images import \
    _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import get_pylo_rule_from_string, is_pylo_rule_recursive

sns.set(style="whitegrid")


def plot_combined_group_info_known_prop_scores_with_pca_selection_confidence_evolution_less_detailed(
        dataset_name: str,
        target_relation: str,
        filter_relation: str,
        propensity_score_subjects_of_filter_relation: float,
        propensity_score_other_entities_list: List[float],
        include_recursive_rules: bool = False
):
    is_pca_version: bool = True
    df_group_info: pd.DataFrame = get_dataframe_with_info_about_known_prop_scores_sar_two_groups__for(
        dataset_name=dataset_name,
        target_relation=target_relation,
        filter_relation=filter_relation,
        propensity_score_subjects_of_filter_relation=propensity_score_subjects_of_filter_relation,
        propensity_score_other_entities_list=propensity_score_other_entities_list,
        is_pca_version=is_pca_version
    )
    list_of_rules: List[str] = list(df_group_info["Rule"].unique())

    group_order: List[str] = GroupNameEnum.get_groups_as_ordered_strings()
    group_color_palette: Dict[str, str] = GroupNameEnum.get_color_palette()

    #################################
    root_dir_experiment_settings: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_experiment_high_level(
        dataset_name=dataset_name,
        target_relation=target_relation,
        filter_relation=filter_relation,
        is_pca_version=is_pca_version
    )
    image_dir: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_images(
        use_pca=is_pca_version, dataset_name=dataset_name,
        scar_propensity_score=propensity_score_subjects_of_filter_relation
    )
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    specific_image_dir: str = os.path.join(
        image_dir,
        'combo_group_info_pca_selection_conf_evol_less_detailed'
    )

    print(f"Writing images to {specific_image_dir}")
    if not os.path.exists(specific_image_dir):
        os.makedirs(specific_image_dir)

    filename_root: str = f"combo_group_info_pca_selection_known_prop_scores_sar" \
                         f"_{target_relation}" \
                         f"_{filter_relation}"

    df_rule_wrappers: pd.DataFrame = _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups(
        root_dir_experiment_settings=root_dir_experiment_settings,
        target_relation=target_relation,
        filter_relation=filter_relation,
        filter_group_prop_score=propensity_score_subjects_of_filter_relation
    )

    column_names_logistics: List[str] = [
        'target_relation',
        'filter_relation',
        'prop_score_subj',
        'prop_score_other',
        'random_trial_index',
        'Rule',
    ]

    ####################################

    rule_to_select: str
    for rule_to_select in list_of_rules:

        pylo_rule: PyloClause = get_pylo_rule_from_string(rule_to_select)
        is_rule_recursive = is_pylo_rule_recursive(pylo_rule)

        if include_recursive_rules or (not include_recursive_rules and not is_rule_recursive):
            melted_df_info_absolute_conf: MeltedDataFrameInfo = get_df_abs_conf_per_group(df_group_info)
            melted_df_info_relative_conf: MeltedDataFrameInfo = get_df_rel_conf_per_group(df_group_info)
            melted_df_info_absolute_n_predictions: MeltedDataFrameInfo = get_df_abs_n_predictions_per_group(df_group_info)
            melted_df_info_relative_n_predictions: MeltedDataFrameInfo = get_df_rel_n_predictions_per_group(df_group_info)

            melted_df_info_absolute_pair_pos_conf_s_to_o: MeltedDataFrameInfo = get_df_abs_pair_pos_conf_s_to_o_per_group(
                df_group_info)

            df_group_info_single_rule = df_group_info[df_group_info["Rule"] == rule_to_select]

            df_absolute_pair_pos_conf_s_to_o_single_rule = melted_df_info_absolute_pair_pos_conf_s_to_o.df[
                melted_df_info_absolute_pair_pos_conf_s_to_o.df["Rule"] == rule_to_select
                ]

            df_absolute_conf_single_rule: pd.DataFrame = melted_df_info_absolute_conf.df[
                (melted_df_info_absolute_conf.df["Rule"] == rule_to_select)
            ]
            df_relative_conf_single_rule: pd.DataFrame = melted_df_info_relative_conf.df[
                melted_df_info_relative_conf.df["Rule"] == rule_to_select
                ]
            df_absolute_n_predictions_single_rule = melted_df_info_absolute_n_predictions.df[
                melted_df_info_absolute_n_predictions.df["Rule"] == rule_to_select

                ]
            df_relative_n_predictions_single_rule = melted_df_info_relative_n_predictions.df[
                melted_df_info_relative_n_predictions.df["Rule"] == rule_to_select
                ]
            ##################################################################################
            # %% md
            # PCA selection mechanism included

            df_conf_comp_true_conf_s_to_o = df_rule_wrappers[
                column_names_logistics +
                list(
                    map(lambda conf: conf.get_name(),
                        [ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O] + ConfidenceEnum.get_estimators_of(
                            ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O
                        )
                        )
                )
                ]
            # df_conf_comp_true_conf_o_to_s = df_rule_wrappers[
            #     column_names_logistics +
            #     list(
            #         map(lambda conf: conf.get_name(),
            #             [ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S] + ConfidenceEnum.get_estimators_of(
            #                 ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S)
            #             )
            #     )
            #     ]
            #####################################################################
            # Combined plot

            fig_width = 8
            fig_height = 3.3
            figsize = (fig_width, fig_height)

            fig = plt.figure(figsize=figsize)
            gs0 = gridspec.GridSpec(2, 2, figure=fig)

            ax_to_plot_abs_group_local_conf = fig.add_subplot(gs0[0, 0])
            ax_to_plot_n_preds_per_group = fig.add_subplot(gs0[1, 0])
            ax_to_plot_conf_evol = fig.add_subplot(gs0[:, 1])

            fig: Figure
            axes: Axes
            # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=False)
            ###############################################################################################################
            ##################################################################
            # Confidence plots
            ##################################################################

            abs_pair_pos_conf_total_group: float = df_absolute_pair_pos_conf_s_to_o_single_rule[
                (df_absolute_pair_pos_conf_s_to_o_single_rule["prediction_group"] == GroupNameEnum.total.value)
            ][melted_df_info_absolute_pair_pos_conf_s_to_o.cname_value_column].iloc[0]

            ax_abs_pair_pos_conf = sns.barplot(
                data=df_absolute_pair_pos_conf_s_to_o_single_rule,
                x=melted_df_info_absolute_pair_pos_conf_s_to_o.cname_value_column,
                y=melted_df_info_absolute_pair_pos_conf_s_to_o.cname_prediction_group,
                hue_order=group_order,
                palette=group_color_palette,
                ax=ax_to_plot_abs_group_local_conf,

            )
            ax_abs_pair_pos_conf.grid(False)
            show_values_on_bars(ax_abs_pair_pos_conf, h_v='h')
            ax_abs_pair_pos_conf.set_xlim([0.0, 1.0])
            ax_abs_pair_pos_conf.set_xticks([0.0, 1.0])

            s_less_conf_val = ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O.value.replace("(S)", "")

            ax_abs_pair_pos_conf.set_xlabel(f"{s_less_conf_val} per group")
            ax_abs_pair_pos_conf.axvline(x=abs_pair_pos_conf_total_group, color='k', linestyle='--')
            ax_abs_pair_pos_conf.set_ylabel('')
            ##################################################################

            abs_conf_total_group: float = df_absolute_conf_single_rule[
                (df_absolute_conf_single_rule["prediction_group"] == GroupNameEnum.total.value)
            ]['absolute_confidence_value'].iloc[0]
            #
            # ax_abs_conf: Axes = sns.barplot(
            #     data=df_absolute_conf_single_rule,
            #     x=melted_df_info_absolute_conf.cname_value_column,
            #     y=melted_df_info_absolute_conf.cname_prediction_group,
            #     hue_order=group_order,
            #     palette=group_color_palette,
            #     ax=ax_to_plot_abs_group_local_conf
            # )
            # show_values_on_bars(ax_abs_conf, h_v='h')
            # ax_abs_conf.set_xlim([0.0, 1.0])
            # ax_abs_conf.set_xlabel("$conf(R \mid group)$")
            # ax_abs_conf.axvline(x=abs_conf_total_group, color='k', linestyle='--')
            # ax_abs_conf.set_ylabel('')

            ##################################################################
            #
            # ax_rel_conf: Axes = sns.barplot(
            #     data=df_relative_conf_single_rule,
            #     x=melted_df_info_relative_conf.cname_value_column,
            #     y=melted_df_info_relative_conf.cname_prediction_group,
            #     palette=group_color_palette,
            #     ax=axes[1, 0]
            # )
            # ax_rel_conf.axvline(x=0.0, color='k', linestyle='--')
            # ax_rel_conf.set_xlim([-1.1, 1.1])
            # ax_rel_conf.set_xlabel('$\\frac{conf(R \mid group) - conf(R)}{conf(R)}$')
            # show_values_on_bars(ax_rel_conf, h_v='h')
            ###################################################################
            ##################################################################
            # Nb of predictions plots
            ##################################################################
            ax_abs_n_preds = sns.barplot(
                data=df_absolute_n_predictions_single_rule,
                x=melted_df_info_absolute_n_predictions.cname_value_column,
                y=melted_df_info_absolute_n_predictions.cname_prediction_group,
                order=group_order,
                palette=group_color_palette,
                ax=ax_to_plot_n_preds_per_group
            )
            ax_abs_n_preds.grid(False)

            ax_abs_n_preds.set_xlabel('abs # predictions per group')
            ax_abs_n_preds.set_ylabel('')
            show_values_on_bars(ax_abs_n_preds, h_v='h', float_precision=0)

            ##################################################################
            #
            # ax_rel_n_preds = sns.barplot(
            #     data=df_relative_n_predictions_single_rule,
            #     x=melted_df_info_relative_n_predictions.cname_value_column,
            #     y=melted_df_info_relative_n_predictions.cname_prediction_group,
            #     palette=group_color_palette,
            #     ax=axes[1, 1]
            # )
            # ax_rel_n_preds.set_xlabel('% # predictions / group')
            # ax_rel_n_preds.set_xlim([0.0, 101])
            # ax_rel_n_preds.axvline(x=100, color='k', linestyle='--')
            # show_values_on_bars(ax_rel_n_preds, h_v='h')
            #
            ##############################################################################################################
            ##################################################################
            # PCA selection confidence plots
            ##################################################################

            value_true_pair_pos_s_to_o_in_filter = df_group_info_single_rule[
                CNameEnum.cname_true_pos_pair_conf_s_to_o_in_filter.value
            ].iloc[0]
            value_true_pair_pos_s_to_o_not_in_filter = df_group_info_single_rule[
                CNameEnum.cname_true_pos_pair_conf_s_to_o_not_in_filter.value
            ].iloc[0]

            df_conf_comp_true_conf_s_to_o_single_rule = df_conf_comp_true_conf_s_to_o[
                df_conf_comp_true_conf_s_to_o["Rule"] == rule_to_select
                ]

            plot_known_prop_score_pca_selection_for_true_conf_star_on_axis(
                df_conf_comp_single_rule=df_conf_comp_true_conf_s_to_o_single_rule,
                scar_propensity_score=propensity_score_subjects_of_filter_relation,
                column_names_logistics=column_names_logistics,
                true_conf_star=ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O,
                ax_conf_value_evolution=ax_to_plot_conf_evol,
                ax_conf_error_evolution=None,

                value_true_conf_total=abs_conf_total_group,
                value_true_conf_star_in_filter=value_true_pair_pos_s_to_o_in_filter,
                value_true_conf_star_not_in_filter=value_true_pair_pos_s_to_o_not_in_filter

            )

            # value_true_pair_pos_o_to_s_in_filter = df_group_info_single_rule[
            #     CNameEnum.cname_true_pos_pair_conf_o_to_s_in_filter.value
            # ].iloc[0]
            # value_true_pair_pos_o_to_s_not_in_filter = df_group_info_single_rule[
            #     CNameEnum.cname_true_pos_pair_conf_o_to_s_not_in_filter.value
            # ].iloc[0]
            #
            # df_conf_comp_true_conf_o_to_s_single_rule = df_conf_comp_true_conf_o_to_s[
            #     df_conf_comp_true_conf_o_to_s["Rule"] == rule_to_select
            #     ]
            # plot_known_prop_score_pca_selection_for_true_conf_star_on_axis(
            #     df_conf_comp_single_rule=df_conf_comp_true_conf_o_to_s_single_rule,
            #     scar_propensity_score=propensity_score_subjects_of_filter_relation,
            #     column_names_logistics=column_names_logistics,
            #     true_conf_star=ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S,
            #     ax_conf_value_evolution=axes[0, 3],
            #     ax_conf_error_evolution=axes[1, 3],
            #
            #     value_true_conf_total=abs_conf_total_group,
            #     value_true_conf_star_in_filter=value_true_pair_pos_o_to_s_in_filter,
            #     value_true_conf_star_not_in_filter=value_true_pair_pos_o_to_s_not_in_filter
            #
            # )
            paper_like_rule_string: str = get_paper_like_rule_string_from_prolog_str(rule_to_select)
            # plt.suptitle(paper_like_rule_string, y=1.03)
            fig.suptitle(f"{paper_like_rule_string} (with $q=" + filter_relation + "$)", y=1.03)
            plt.tight_layout()
            # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename_image_combo: str = os.path.join(
                specific_image_dir,
                filename_root + f"_{rule_to_select}.png"
            )

            plt.savefig(filename_image_combo, bbox_inches='tight')
            plt.close(fig)

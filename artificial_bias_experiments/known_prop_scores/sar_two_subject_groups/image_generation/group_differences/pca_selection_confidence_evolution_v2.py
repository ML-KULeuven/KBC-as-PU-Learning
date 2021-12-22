import math
import os
from typing import List, Optional

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

#
# def pca_selection_mechanism_known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule(
#         df_rule_wrappers: pd.DataFrame,
#         filter_relation: str,
#         image_dir: str,
#         filename_root: str,
#         scar_propensity_score: float
#
# ) -> None:
#     dir_conf_star_vs_pca_estimators: str = os.path.join(
#         image_dir,
#         "pca_selection_confidence_evolution_true_conf_star_vs_pca_estimators"
#     )
#     if not os.path.exists(dir_conf_star_vs_pca_estimators):
#         os.makedirs(dir_conf_star_vs_pca_estimators)
#
#     column_names_logistics: List[str] = [
#         'target_relation',
#         'filter_relation',
#         'prop_score_subj',
#         'prop_score_other',
#         'random_trial_index',
#         'Rule',
#     ]
#
#     for true_conf in [ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O, ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S]:



# def foobar(
#         df_rule_wrappers: pd.DataFrame,
#         filter_relation: str,
#         image_dir: str,
#         scar_propensity_score: float
# ) -> None:
#     dir_conf_star_vs_pca_estimators: str = os.path.join(
#         image_dir,
#         "pca_selection_confidence_evolution_true_conf_star_vs_pca_estimators"
#     )
#     if not os.path.exists(dir_conf_star_vs_pca_estimators):
#         os.makedirs(dir_conf_star_vs_pca_estimators)
#
#     column_names_logistics: List[str] = [
#         'target_relation',
#         'filter_relation',
#         'prop_score_subj',
#         'prop_score_other',
#         'random_trial_index',
#         'Rule',
#     ]
#
#     df_conf_comp_true_conf_s_to_o = df_rule_wrappers[
#         column_names_logistics +
#         list(
#             map(lambda conf: conf.get_name(),
#                 [ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O] + ConfidenceEnum.get_estimators_of(ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O)
#              )
#         )
#     ]
#     df_conf_comp_true_conf_o_to_s = df_rule_wrappers[
#         column_names_logistics +
#         list(
#             map(lambda conf: conf.get_name(),
#                 [ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S] + ConfidenceEnum.get_estimators_of(ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S)
#              )
#         )
#     ]
#
#     rule_name: str
#     for rule_name in df_rule_wrappers["Rule"].unique():
#         df_conf_comp_true_conf_s_to_o_single_rule = df_conf_comp_true_conf_s_to_o[
#             df_conf_comp_true_conf_s_to_o["Rule"] == rule_name
#         ]
#         plot_known_prop_score_pca_selection_for_true_conf_star_on_axis(
#             df_conf_comp_single_rule=df_conf_comp_true_conf_s_to_o_single_rule,
#             scar_propensity_score=scar_propensity_score,
#             column_names_logistics=column_names_logistics,
#             true_conf_star=ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O,
#             ax_conf_value_evolution=,
#             ax_conf_error_evolution=
#         )
#
#         df_conf_comp_true_conf_o_to_s_single_rule = df_conf_comp_true_conf_o_to_s[
#             df_conf_comp_true_conf_o_to_s["Rule"] == rule_name
#         ]
#         plot_known_prop_score_pca_selection_for_true_conf_star_on_axis(
#             df_conf_comp_single_rule=df_conf_comp_true_conf_o_to_s_single_rule,
#             scar_propensity_score=scar_propensity_score,
#             column_names_logistics=column_names_logistics,
#             true_conf_star=ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S,
#             ax_conf_value_evolution=,
#             ax_conf_error_evolution=
#         )

def plot_known_prop_score_pca_selection_for_true_conf_star_on_axis(
        df_conf_comp_single_rule: pd.DataFrame,
        # n_random_trials: str,
        scar_propensity_score: float,
        column_names_logistics: List[str],
        true_conf_star: ConfidenceEnum,

        value_true_conf_star_in_filter: float,
        value_true_conf_star_not_in_filter: float,


        ax_conf_value_evolution: Optional[Axes]=None,
        ax_conf_error_evolution: Optional[Axes]=None,
        should_plot_true_conf_total: bool = False,

        value_true_conf_total: Optional[float] = None,
        target_relation = None,
        filter_relation = None,

):

    if ax_conf_value_evolution is not None or ax_conf_error_evolution is not None:

        conf_estimators_to_compare = ConfidenceEnum.get_estimators_of(true_conf_star)
        confs_comparison = [true_conf_star] + conf_estimators_to_compare

        column_names_info = ColumnNamesInfo(
            true_conf=true_conf_star,
            column_name_true_conf=true_conf_star.get_name(),
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

        df_conf_comp_melted = df_conf_comp_single_rule.melt(
            id_vars=column_names_logistics,
            var_name='confidence_type',
            value_name='confidence_value'
        )

        df_diff_to_true_conf_melted: pd.DataFrame = get_df_diffs_between_true_conf_and_confidence_estimators_melted(
            df_rule_wrappers=df_conf_comp_single_rule,
            column_names_info=column_names_info
        )

        added_rows = []
        for prop_score_other in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            new_row = [
                target_relation, filter_relation,
                scar_propensity_score,
                prop_score_other,
                0,  # random trial index
                "rule",
                f"{ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O.value} (" + "$S_{q}$" + ")",
                value_true_conf_star_in_filter

            ]
            added_rows.append(new_row)
            new_row = [
                target_relation, filter_relation,
                scar_propensity_score,
                prop_score_other,
                0,  # random trial index
                "rule",
                f"{ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O.value} (" + "$S_{\\neg q}$" + ")",
                value_true_conf_star_not_in_filter

            ]
            added_rows.append(new_row)
        new_df = pd.DataFrame(data=added_rows, columns=df_conf_comp_melted.columns)
        df_conf_comp_melted = pd.concat([df_conf_comp_melted, new_df], axis=0)

        error_metric_label: str = get_confidence_difference_label_string(true_conf_star)

        conf_max = df_conf_comp_melted['confidence_value'].max()
        conf_min = df_conf_comp_melted['confidence_value'].min()
        conf_scar_y_pos = conf_min + (conf_max-conf_min)/2



        color_palette[f"{ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O.value} (" + "$S_{q}$" + ")"] = "green"
        color_palette[f"{ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O.value} (" + "$S_{\\neg q}$" + ")"] = "red"


        if ax_conf_value_evolution is not None:

            fig: Figure
            axes: Axes
            ax_confs: Axes = sns.lineplot(
                x="prop_score_other",
                y='confidence_value',
                hue="confidence_type",
                # hue_order=order,
                palette=color_palette,
                # marker="o",
                data=df_conf_comp_melted,
                ax=ax_conf_value_evolution,
            )
            sns.despine(ax=ax_confs, offset=10)

            ax_confs.set_xlim([0.0, 1.0])
            ax_confs.set_xticks([0.0, 0.5, 1.0])
            # ax_confs.set_ylim([0.0, None])
            ax_confs.set_xlabel('$c_{\\neg q}$')
            ax_confs.set_ylabel("$\widehat{conf}(R)$")
            # ax_confs.set_title(
            #     "$\widehat{conf^{*}}(R)$",
            #     loc='left'
            #     # f" (Avg over {n_random_trials} random selections)\n"
            # )
            ax_confs.grid(False)

            conf_diff = conf_max - conf_min

            ax_confs.set_ylim(conf_min - conf_diff/20, conf_max+conf_diff/20)

            # ax_confs.axhline(y=value_true_conf_star_in_filter, color='g')
            # ax_confs.text(x=0.1, y=value_true_conf_star_in_filter, color='g', s='$conf^{*}(R \mid  q) $')
            #
            # ax_confs.axhline(y=value_true_conf_star_not_in_filter,color='r')
            # ax_confs.text(x=0.1, y=value_true_conf_star_not_in_filter,color='r', s='$conf^{*}(R \mid \\neg q)$')

            if should_plot_true_conf_total:
                if value_true_conf_total is None:
                    raise Exception("No value given for the true confidence")
                else:
                    ax_confs.axhline(y=value_true_conf_total, color='xkcd:magenta')
                    ax_confs.text(x=0.1, y=value_true_conf_total, color='xkcd:magenta', s='conf(R)')

            ymin, ymax = ax_confs.get_ylim()
            xmin, xmax = ax_confs.get_xlim()

            # ax_confs.legend(bbox_to_anchor=(-0.8, 1.0), loc='upper left')
            ax_confs.axvline(x=scar_propensity_score, color='k', linestyle='--')
            # ax_confs.text(x=scar_propensity_score, y=(ymax-ymin)/2+ymin, s="SCAR", verticalalignment='center')

            ax_confs.axhspan(
                xmin=xmin,
                xmax=0.5,
                ymin=ymin,
                ymax=ymax,
                facecolor='#009900',
                alpha=0.2
            )

            ax_confs.axhspan(
                xmin=0.5,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                facecolor='#F80A0A',
                alpha=0.2
            )

            handles, labels = ax_confs.get_legend_handles_labels()
            from matplotlib.legend import Legend
            ax_confs_tmp_legend: Legend = ax_confs.get_legend()
            legend_rectangle = ax_confs_tmp_legend.get_frame()
            legend_width = legend_rectangle.get_width()
            labels = [lab.replace("(S)", "") for lab in labels]


            ax_confs.legend(
                handles=handles[1:], labels=labels[1:],
                bbox_to_anchor=(1, 0.5),
                loc='center left',
                title='confidence')

        ##########################################
        if ax_conf_error_evolution is not None:
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
                ax=ax_conf_error_evolution
            )
            # ax_confs.legend_ = leg

            ax_squared_error.set_xlim([0.0, 1.0])
            # ax_squared_error.set_ylim([0.0, None])
            ax_squared_error.set_xlabel('$c_{\\neg q}$')
            ax_squared_error.set_ylabel(error_metric_label)
            ax_squared_error.set_title(
                f"{error_metric_label}"
                # f" (Avg over {n_random_trials} random selections)\n"
            )


            ymin, ymax = ax_squared_error.get_ylim()
            xmin, xmax = ax_squared_error.get_xlim()

            ax_squared_error.axhspan(
                xmin=xmin,
                xmax=0.5,
                ymin=ymin,
                ymax=ymax,
                facecolor='#009900',
                alpha=0.2
            )

            ax_squared_error.axhspan(
                xmin=0.5,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                facecolor='#F80A0A',
                alpha=0.2
            )

            ax_squared_error.axvline(x=scar_propensity_score, color='k', linestyle='--')
            ax_squared_error.text(x=scar_propensity_score, y=(ymax-ymin)/2+ymin, s="SCAR", verticalalignment='center')


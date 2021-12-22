import os
from typing import List, NamedTuple, Dict

import pandas as pd

from artificial_bias_experiments.evaluation.confidence_comparison.df_utils import ColumnNamesInfo, \
    get_df_diffs_between_true_conf_and_confidence_estimators_melted
from artificial_bias_experiments.image_generation.image_generation_utils import \
    get_confidence_difference_label_string, get_confidence_estimate_label_string
from kbc_pul.confidence_naming import ConfidenceEnum

import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.axes import Axes
from matplotlib.figure import Figure

class ConfCompPlotInfo(NamedTuple):
    df_conf_comp_melted: pd.DataFrame
    n_rules: int
    n_random_trials: int
    color_palette: Dict[str, str]


def get_confidence_comparison_plot_info(
        confidences_to_compare: List[ConfidenceEnum],
        column_names_logistics: List[str],
        df_rule_wrappers: pd.DataFrame,
) -> ConfCompPlotInfo:
    column_names_confidence_comparison = [
        conf.get_name()
        for conf in confidences_to_compare
    ]

    df_conf_comp = df_rule_wrappers[
        column_names_logistics + column_names_confidence_comparison
        ]
    df_conf_comp_melted = df_conf_comp.melt(
        id_vars=column_names_logistics,
        var_name='confidence_type',
        value_name='confidence_value'
    )
    n_rules = len(df_conf_comp["Rule"].unique())
    n_random_trials = len(df_conf_comp["random_trial_index"].unique())
    color_palette = {
        conf.get_name(): conf.get_hex_color_str()
        for conf in confidences_to_compare
    }
    return ConfCompPlotInfo(
        df_conf_comp_melted=df_conf_comp_melted,
        n_rules=n_rules,
        n_random_trials=n_random_trials,
        color_palette=color_palette
    )


def generate_image_confidence_estimates(
        true_conf: ConfidenceEnum,
        df_rule_wrappers: pd.DataFrame,
        n_random_trials: int,
        column_names_info: ColumnNamesInfo,
        image_dir: str,
        filename_root: str
) -> None:

    dir_confidence_estimates_images: str = os.path.join(
        image_dir,
        "confidence_estimates"
    )
    if not os.path.exists(dir_confidence_estimates_images):
        os.makedirs(dir_confidence_estimates_images)

    rule_name: str
    for rule_name in df_rule_wrappers["Rule"].unique():

        df_single_rule_wrapper = df_rule_wrappers[df_rule_wrappers["Rule"] == rule_name]

        df_rule_wrappers_melted = df_single_rule_wrapper.melt(
            id_vars=column_names_info.column_names_logistics,
            var_name='confidence_type',
            value_name='confidence_value'
        )

        filename_image_confidence_estimates: str = os.path.join(
            dir_confidence_estimates_images,
            f"{filename_root}_conf_estimates_{rule_name}.png"
        )

        color_palette: Dict[str, str] = column_names_info.get_conf_estimators_color_palette()
        color_palette[true_conf.get_name()] = true_conf.get_hex_color_str()

        vertical_axis_label: str = get_confidence_estimate_label_string(true_conf)

        figsize = (10, 5)

        fig: Figure
        axes: Axes
        fig, axes = plt.subplots(1, 1, figsize=figsize)

        ax: Axes = sns.lineplot(
            x="label_frequency",
            y='confidence_value',
            hue="confidence_type",
            # hue_order=order,
            palette=color_palette,
            marker="o",
            data=df_rule_wrappers_melted,
        )

        ax.set_xlim([0, 1])
        ax.set_ylim([0, None])

        ax.set_xlabel(f'label frequency c')
        ax.set_ylabel(vertical_axis_label)
        ax.set_title("" + vertical_axis_label +
                     # f" for {rule_name}"
                     f" (Avg over {n_random_trials} random selections)"
        )

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        plt.suptitle(f"SCAR for {rule_name}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename_image_confidence_estimates, bbox_inches='tight')
        plt.close(fig)


def plot_pca_confidence_behavior(
        column_names_logistics: List[str],
        df_rule_wrappers: pd.DataFrame,
        target_relation: str,
        filename_image: str
):
    true_conf = ConfidenceEnum.TRUE_CONF

    conf_to_compare_to_pca_comparison = [
        ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O,
        ConfidenceEnum.PCA_CONF_S_TO_O,

        ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S,
        ConfidenceEnum.PCA_CONF_O_TO_S
    ]

    confs_pca_comparison = [ConfidenceEnum.TRUE_CONF] + conf_to_compare_to_pca_comparison
    column_names_pca_confidence_comparison = [
        conf.get_name()
        for conf in confs_pca_comparison
    ]

    df_pca_conf_comp = df_rule_wrappers[
        column_names_logistics + column_names_pca_confidence_comparison
        ]

    df_pca_conf_comp_melted = df_pca_conf_comp.melt(
        id_vars=column_names_logistics,
        var_name='confidence_type',
        value_name='confidence_value'
    )
    n_rules = len(df_pca_conf_comp["Rule"].unique())
    n_random_trials = len(df_pca_conf_comp["random_trial_index"].unique())

    column_names_info = ColumnNamesInfo(
        true_conf=true_conf,
        column_name_true_conf=true_conf.get_name(),
        conf_estimators=conf_to_compare_to_pca_comparison,
        column_names_conf_estimators=[
            col.get_name()
            for col in conf_to_compare_to_pca_comparison
        ],
        column_names_logistics=column_names_logistics
    )

    df_diff_to_true_conf_melted: pd.DataFrame = get_df_diffs_between_true_conf_and_confidence_estimators_melted(
        df_rule_wrappers=df_pca_conf_comp,
        column_names_info=column_names_info
    )

    color_palette = {
        conf.get_name(): conf.get_hex_color_str()
        for conf in confs_pca_comparison
    }
    error_metric_label: str = get_confidence_difference_label_string(true_conf)

    figsize = (10, 5)

    fig: Figure
    axes: Axes
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    min_error = df_diff_to_true_conf_melted["squared_error"].min()
    max_error = df_diff_to_true_conf_melted["squared_error"].max()

    cwa_vertical_position = (min_error + max_error) / 2

    ax_confs: Axes = sns.lineplot(
        x="label_frequency",
        y='confidence_value',
        hue="confidence_type",
        # hue_order=order,
        palette=color_palette,
        marker="o",
        data=df_pca_conf_comp_melted,
        ax=axes[0]
    )
    ax_confs.set_xlabel(f'label frequency c')
    ax_confs.set_ylabel("$\widehat{conf}(R)$")
    ax_confs.set_title("SCAR: $\widehat{conf}(R)$\n"
                       f" for {n_rules} rules predicting {target_relation}\n"
                       f" (Avg over rules & {n_random_trials} random selections)\n"
                       )
    ax_confs.legend(bbox_to_anchor=(-0.3, 1.0), loc='upper right')
    # ax_confs.get_legend().remove()

    ax_squared_error: Axes = sns.lineplot(
        x="label_frequency",
        y='squared_error',
        hue="error_type",
        # hue_order=order,
        palette=color_palette,
        marker="o",
        data=df_diff_to_true_conf_melted,
        ax=axes[1]
    )

    ax_squared_error.set_xlabel(f'label frequency c')
    ax_squared_error.set_ylabel(error_metric_label)
    ax_squared_error.set_title("SCAR: " + error_metric_label + "\n"
                                                               f" for {n_rules} rules predicting {target_relation}\n"
                                                               f" (Avg over {n_rules} rules & {n_random_trials} random selections)\n"
                               )
    ax_squared_error.get_legend().remove()

    plt.axvline(x=1.0, color='k', linestyle='--')
    plt.text(x=1.0, y=cwa_vertical_position, s="CWA", verticalalignment='center')
    plt.tight_layout()
    plt.savefig(filename_image, bbox_inches='tight')
    plt.close(fig)


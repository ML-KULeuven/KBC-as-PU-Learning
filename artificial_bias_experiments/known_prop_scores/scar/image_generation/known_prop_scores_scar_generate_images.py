import os
import re
from pathlib import Path
from typing import List, Match, Optional, Set, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from artificial_bias_experiments.evaluation.confidence_comparison.df_utils import \
    get_df_rulewise_fraction_of_conf_estimator_to_true_conf, ColumnNamesInfo, \
    get_df_diffs_between_true_conf_and_confidence_estimators_melted
from artificial_bias_experiments.image_generation.image_generation_utils import \
    get_confidence_difference_label_string, get_confidence_fraction_label_string, get_confidence_estimate_label_string
from artificial_bias_experiments.known_prop_scores.scar.image_generation.confidence_evolution import \
    generate_image_confidence_estimates
from artificial_bias_experiments.known_prop_scores.scar.image_generation.confidence_evolution_std_and_inverse_e import \
    known_prop_scores_scar_plot_conf_evolution_true_conf_vs_std_and_inverse_e_per_rule
from artificial_bias_experiments.known_prop_scores.scar.image_generation.confidence_evolution_pca_estimators import \
    known_prop_scores_scar_plot_conf_evolution_true_conf_vs_pca_estimators_per_rule
from artificial_bias_experiments.known_prop_scores.scar.known_prop_scores_scar_file_naming import \
    KnownPropScoresSCARConstantLabelFreqFileNamer

from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import RuleWrapper

sns.set(style="whitegrid")

re_label_frequency_pattern = re.compile(r"c([0-1]\.?[0-9]*)")
re_trial_pattern = re.compile(r"trial(\d+)")


def _get_label_frequency_from_string(input_string: str) -> float:
    o_match: Optional[Match[str]] = re.search(re_label_frequency_pattern, input_string)
    if o_match is not None:
        parsed_label_frequency = float(o_match.group(1))
        return parsed_label_frequency
    else:
        raise Exception(f"No label frequency found in string {input_string}")


def _get_trial_index_from_string(input_string: str) -> int:
    o_match: Optional[Match[str]] = re.search(re_trial_pattern, input_string)
    if o_match is not None:
        random_trial_index = int(o_match.group(1))
        return random_trial_index
    else:
        raise Exception(f"No propensity scores found in string {input_string}")


def _get_rule_wrapper_row_scar(
        rule_wrapper: RuleWrapper,
        target_relation: str,
        true_label_frequency: float,
        random_trial_index
):
    partial_row: List[str] = [
        target_relation,
        true_label_frequency,
        random_trial_index
    ]
    partial_row_rule_wrapper: List[str] = rule_wrapper.to_row(include_amie_metrics=False)
    full_row = partial_row + partial_row_rule_wrapper
    return full_row


def _get_rule_wrappers_as_dataframe(
        root_dir_experiment_settings: str,
        target_relation: str
) -> pd.DataFrame:
    path_dir_root: Path = Path(root_dir_experiment_settings)

    column_names: List[str] = [
                                  'target_relation',
                                  'label_frequency',
                                  'random_trial_index'
                              ] + RuleWrapper.get_columns_header_without_amie()

    data_all_wrappers: List[List] = []
    for descendant_path in path_dir_root.glob('**/*'):
        if descendant_path.is_file() and descendant_path.suffix in {".gz"}:
            current_rule_wrapper: RuleWrapper = RuleWrapper.read_json(str(descendant_path))
            current_label_frequency: float = _get_label_frequency_from_string(str(descendant_path))
            random_trial_index: int = _get_trial_index_from_string(str(descendant_path))
            data_all_wrappers.append(
                _get_rule_wrapper_row_scar(
                    rule_wrapper=current_rule_wrapper,
                    target_relation=target_relation,
                    true_label_frequency=current_label_frequency,
                    random_trial_index=random_trial_index
                )
            )
    df_rule_wrappers: pd.DataFrame = pd.DataFrame(data_all_wrappers, columns=column_names)
    return df_rule_wrappers




def generate_image_confidence_difference(
        true_conf: ConfidenceEnum,
        df_diff_to_true_conf_melted: pd.DataFrame,
        target_relation: str,
        n_rules: int,
        n_random_trials: int,
        column_names_info: ColumnNamesInfo,
        filename_image: str
) -> None:

    color_palette: Dict[str, str] = column_names_info.get_conf_estimators_color_palette()

    error_metric_label: str = get_confidence_difference_label_string(true_conf)

    figsize = (10, 5)

    fig: Figure
    axes: Axes
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    min_error = df_diff_to_true_conf_melted["squared_error"].min()
    max_error = df_diff_to_true_conf_melted["squared_error"].max()

    cwa_vertical_position = (min_error + max_error) / 2

    ax: Axes = sns.lineplot(
        x="label_frequency",
        y='squared_error',
        hue="error_type",
        # hue_order=order,
        palette=color_palette,
        marker="o",
        data=df_diff_to_true_conf_melted,
    )

    ax.set_xlabel(f'label frequency c')
    ax.set_ylabel(error_metric_label)
    ax.set_title("SCAR: Avg " + error_metric_label +
                 f" for {n_rules} rules predicting {target_relation}\n"
                 f" (Avg over {n_rules} rules & {n_random_trials} random selections)\n"
    )

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.axvline(x=1.0, color='k', linestyle='--')
    plt.text(x=1.0, y=cwa_vertical_position, s="CWA", verticalalignment='center')

    plt.tight_layout()
    plt.savefig(filename_image, bbox_inches='tight')
    plt.close(fig)


def generate_image_confidence_fraction(
        true_conf: ConfidenceEnum,
        df_fractions_to_true_conf_melted: pd.DataFrame,
        target_relation: str,
        n_rules: int,
        n_random_trials: int,
        column_names_info: ColumnNamesInfo,
        filename_image: str
) -> None:
    color_palette: Dict[str, str] = column_names_info.get_conf_estimators_color_palette()

    min_error = df_fractions_to_true_conf_melted["error_value"].min()
    max_error = df_fractions_to_true_conf_melted["error_value"].max()

    error_metric_label: str = get_confidence_fraction_label_string(true_conf)

    cwa_vertical_position = (min_error + max_error) / 2

    figsize = (10, 5)
    fig: Figure
    axes: Axes
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    ax: Axes = sns.lineplot(
        x="label_frequency",
        y='error_value',
        hue="error_type",
        marker="o",
        palette=color_palette,
        data=df_fractions_to_true_conf_melted,
    )
    ax.set_xlabel(f'label frequency c')
    ax.set_ylabel("$\widehat{conf}(R)/conf(R)$")
    ax.set_title("SCAR: Avg $\widehat{conf}(R)/conf(R)$"
                 f" for {n_rules} rules predicting {target_relation}\n"
                 f" (Avg over rules & {n_random_trials} random selections)\n"
                 )

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.axvline(x=1.0, color='k', linestyle='--')
    plt.text(x=1.0, y=cwa_vertical_position, s="CWA", verticalalignment='center')
    plt.tight_layout()
    plt.savefig(filename_image, bbox_inches='tight')
    plt.close(fig)


def generate_images_conf_comparison_known_prop_scores_scar(
        dataset_name: str,
        target_relation: str,
        is_pca_version: bool
):
    # dataset_name: str = "yago3_10"
    # target_relation: str = "haschild"
    # # label_frequency_list: List[float] = [0.2, 0.4, 0.6, 0.8, 1]
    # is_pca_version: bool = False

    if is_pca_version:
        pca_indicator: str = "pca_version"
        true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O
        conf_estimators_to_ignore: Set[ConfidenceEnum] = set()
    else:
        pca_indicator: str = "not_pca"
        true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF
        conf_estimators_to_ignore: Set[ConfidenceEnum] = {ConfidenceEnum.ICW_CONF}

    root_dir_experiment_settings: str = KnownPropScoresSCARConstantLabelFreqFileNamer.get_dir_experiment_high_level(
        dataset_name=dataset_name,
        target_relation=target_relation,
        is_pca_version=is_pca_version
    )

    image_dir: str = KnownPropScoresSCARConstantLabelFreqFileNamer.get_dir_images(
        use_pca=is_pca_version,
        dataset_name=dataset_name,
    )
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    df_rule_wrappers: pd.DataFrame = _get_rule_wrappers_as_dataframe(
        root_dir_experiment_settings=root_dir_experiment_settings,
        target_relation=target_relation
    )

    column_names_logistics: List[str] = [
        'target_relation',
        'label_frequency',
        'random_trial_index',
        "Rule"
    ]

    column_names_info = ColumnNamesInfo.build(
        true_conf=true_conf,
        conf_estimators_to_ignore=conf_estimators_to_ignore,
        column_names_logistics=column_names_logistics
    )

    df_rule_wrappers_relative_to_true_conf = df_rule_wrappers[
        column_names_info.get_df_rule_wrappers_columns_to_use()
    ]

    n_rules = len(df_rule_wrappers_relative_to_true_conf["Rule"].unique())
    n_random_trials = len(df_rule_wrappers_relative_to_true_conf["random_trial_index"].unique())
    filename_root: str = f"known_prop_scores_scar_{target_relation}_{pca_indicator}"

    if true_conf == ConfidenceEnum.TRUE_CONF:
        known_prop_scores_scar_plot_conf_evolution_true_conf_vs_std_and_inverse_e_per_rule(
            df_rule_wrappers=df_rule_wrappers,
            image_dir=image_dir,
            filename_root=filename_root
        )
        known_prop_scores_scar_plot_conf_evolution_true_conf_vs_pca_estimators_per_rule(
            df_rule_wrappers=df_rule_wrappers,
            image_dir=image_dir,
            filename_root=filename_root
        )

    generate_image_confidence_estimates(
        true_conf=true_conf,
        df_rule_wrappers=df_rule_wrappers_relative_to_true_conf,
        n_random_trials=n_random_trials,
        column_names_info=column_names_info,
        image_dir=image_dir,
        filename_root=filename_root
    )

    ########################################################################
    # Calc differences against true conf

    df_diff_to_true_conf_melted: pd.DataFrame = get_df_diffs_between_true_conf_and_confidence_estimators_melted(
        df_rule_wrappers=df_rule_wrappers_relative_to_true_conf,
        column_names_info=column_names_info
    )

    dir_confidence_diff_images: str = os.path.join(
        image_dir,
        "confidence_diffs"
    )
    if not os.path.exists(dir_confidence_diff_images):
        os.makedirs(dir_confidence_diff_images)

    filename_image_confidence_difference: str = os.path.join(
        dir_confidence_diff_images,
        f"{filename_root}_conf_difference.png"
    )
    print(filename_image_confidence_difference)

    generate_image_confidence_difference(
        true_conf=true_conf,
        df_diff_to_true_conf_melted=df_diff_to_true_conf_melted,
        target_relation=target_relation,
        n_rules=n_rules,
        n_random_trials=n_random_trials,
        column_names_info=column_names_info,
        filename_image=filename_image_confidence_difference
    )
    ########################################################################
    # Let us look at fractions

    df_fractions_conf_estimator_to_true_conf: pd.DataFrame = get_df_rulewise_fraction_of_conf_estimator_to_true_conf(
        df_rule_wrappers=df_rule_wrappers_relative_to_true_conf,
        column_names_info=column_names_info
    )

    df_fractions_to_true_conf_melted: pd.DataFrame = df_fractions_conf_estimator_to_true_conf.melt(
        id_vars=column_names_logistics,
        var_name='error_type',
        value_name='error_value'
    )

    dir_confidence_fraction_images: str = os.path.join(
        image_dir,
        "confidence_fraction"
    )
    if not os.path.exists(dir_confidence_fraction_images):
        os.makedirs(dir_confidence_fraction_images)

    filename_image_confidence_fraction: str = os.path.join(
        dir_confidence_fraction_images,
        f"{filename_root}_conf_fraction.png"
    )

    generate_image_confidence_fraction(
        true_conf=true_conf,
        df_fractions_to_true_conf_melted=df_fractions_to_true_conf_melted,
        target_relation=target_relation,
        n_rules=n_rules,
        n_random_trials=n_random_trials,
        column_names_info=column_names_info,
        filename_image=filename_image_confidence_fraction
    )

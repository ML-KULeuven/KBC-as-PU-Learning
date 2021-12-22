import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Match, Optional, Set

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from artificial_bias_experiments.evaluation.confidence_comparison.df_utils import ColumnNamesInfo, \
    get_df_rulewise_fraction_of_conf_estimator_to_true_conf, \
    get_df_diffs_between_true_conf_and_confidence_estimators_melted
from artificial_bias_experiments.image_generation.image_generation_utils import \
    get_confidence_difference_label_string
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.confidence_evolution_pca_estimators import \
    known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.pca_selection.confidence_evolution_pca_selection_mechanism import \
    pca_selection_mechanism_known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.confidence_evolution_std_and_inverse_e import \
    known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_vs_std_and_inverse_e_per_rule
from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.known_prop_scores_sar_two_groups_file_naming import \
    KnownPropScoresSARTwoGroupsFileNamer
from kbc_pul.confidence_naming import ConfidenceEnum
from kbc_pul.data_structures.rule_wrapper import RuleWrapper

sns.set(style="whitegrid")

re_prop_score_pattern = re.compile(r"s_prop([0-1]\.?[0-9]*)_ns_prop([0-1]\.?[0-9]*)")
re_trial_pattern = re.compile(r"trial(\d+)")


def get_rule_wrapper_row_sar_two_groups(
        rule_wrapper: RuleWrapper,
        target_relation: str,
        filter_relation: str,
        true_prop_score_subjects_of_filter_relation: float,
        true_prop_score_other_entities: float,
        random_trial_index
):
    partial_row: List[str] = [
        target_relation,
        filter_relation,
        true_prop_score_subjects_of_filter_relation,
        true_prop_score_other_entities,
        random_trial_index
    ]
    partial_row_rule_wrapper: List[str] = rule_wrapper.to_row(include_amie_metrics=False)
    full_row = partial_row + partial_row_rule_wrapper
    return full_row


def get_propensity_scores_from_string(input_string: str) -> Tuple[float, float]:
    o_match: Optional[Match[str]] = re.search(re_prop_score_pattern, input_string)
    if o_match is not None:
        prop_score_sub = float(o_match.group(1))
        prop_score_not_sub = float(o_match.group(2))
        return prop_score_sub, prop_score_not_sub
    else:
        raise Exception(f"No propensity scores found in string {input_string}")


def get_trial_index_from_string(input_string: str) -> int:
    o_match: Optional[Match[str]] = re.search(re_trial_pattern, input_string)
    if o_match is not None:
        random_trial_index = int(o_match.group(1))
        return random_trial_index
    else:
        raise Exception(f"No random trial index found in string {input_string}")


def get_column_names_df_rule_wrappers_sar_two_groups():
    column_names: List[str] = [
                                  'target_relation',
                                  'filter_relation',
                                  'prop_score_subj',
                                  'prop_score_other',
                                  'random_trial_index'
                              ] + RuleWrapper.get_columns_header_without_amie()
    return column_names


def _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups(
        root_dir_experiment_settings: str,
        target_relation: str,
        filter_relation: str,
        filter_group_prop_score: Optional[float] = None

) -> pd.DataFrame:
    path_dir_root: Path = Path(root_dir_experiment_settings)

    data_all_wrappers: List[List] = []
    for descendant_path in path_dir_root.glob('**/*'):
        if descendant_path.is_file() and descendant_path.suffix in {".gz"}:
            # print(descendant_path)
            current_rule_wrapper: RuleWrapper = RuleWrapper.read_json(str(descendant_path))
            propensity_score_sub, prop_score_other = get_propensity_scores_from_string(str(descendant_path))

            if filter_group_prop_score is None or propensity_score_sub == filter_group_prop_score:
                random_trial_index: int = get_trial_index_from_string(str(descendant_path))
                data_all_wrappers.append(
                    get_rule_wrapper_row_sar_two_groups(
                        rule_wrapper=current_rule_wrapper,
                        target_relation=target_relation,
                        filter_relation=filter_relation,
                        true_prop_score_subjects_of_filter_relation=propensity_score_sub,
                        true_prop_score_other_entities=prop_score_other,
                        random_trial_index=random_trial_index
                    )
                )
    df_rule_wrappers: pd.DataFrame = pd.DataFrame(
        data=data_all_wrappers,
        columns=get_column_names_df_rule_wrappers_sar_two_groups()
    )
    return df_rule_wrappers


def generate_image_confidence_difference(
        true_conf: ConfidenceEnum,
        df_diff_to_true_conf_melted: pd.DataFrame,
        target_relation: str,
        filter_relation: str,
        n_rules: int,
        n_random_trials: int,
        column_names_info: ColumnNamesInfo,
        scar_propensity_score: float,
        prop_score_other_list: List[float],
        filename_image: str
) -> None:
    color_palette: Dict[str, str] = column_names_info.get_conf_estimators_color_palette()

    error_metric_label: str = get_confidence_difference_label_string(true_conf)

    figsize = (10, 5)

    fig: Figure
    axes: Axes
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    cwa_vertical_position = df_diff_to_true_conf_melted['squared_error'].mean()
    prop_score_others_str = "[" + ", ".join(map(str, prop_score_other_list)) + "]"

    ax: Axes = sns.lineplot(
        x="prop_score_other",
        y='squared_error',
        hue="error_type",
        palette=color_palette,
        marker="o",
        data=df_diff_to_true_conf_melted,
    )
    ax.set_xlabel(f'$e(s)$: not {filter_relation}')
    ax.set_ylabel(error_metric_label)

    ax.set_title("SAR: Avg " + error_metric_label +
                 f" for rules predicting p={target_relation}, 2 groups based on q={filter_relation}\n"
                 f" (Avg over {n_rules} rules & {n_random_trials} random selections)\n"
                 "$e((s,p,*) | (s,q,*)\in K) = " + f"{scar_propensity_score}" + "$        "
                                                                                "$e((s,p,*) | \\neg (s,q, *) \in K ) = " + f"{prop_score_others_str} " + "$"
                 )

    plt.axvline(x=scar_propensity_score, color='k', linestyle='--')
    plt.text(x=scar_propensity_score, y=cwa_vertical_position, s="SCAR", verticalalignment='center')

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename_image, bbox_inches='tight')
    plt.close(fig)


def generate_image_confidence_fraction(
        true_conf: ConfidenceEnum,
        df_fractions_to_true_conf_melted: pd.DataFrame,
        target_relation: str,
        filter_relation: str,
        n_rules: int,
        n_random_trials: int,
        column_names_info: ColumnNamesInfo,
        scar_propensity_score: float,
        prop_score_other_list: List[float],
        filename_image: str
) -> None:
    color_palette: Dict[str, str] = column_names_info.get_conf_estimators_color_palette()

    error_metric_label: str = get_confidence_difference_label_string(true_conf)
    prop_score_others_str = "[" + ", ".join(map(str, prop_score_other_list)) + "]"

    figsize = (10, 5)
    fig: Figure
    axes: Axes
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    cwa_vertical_position = df_fractions_to_true_conf_melted['error_value'].mean()

    ax: Axes = sns.lineplot(
        x="prop_score_other",
        y='error_value',
        hue="error_type",
        marker="o",
        palette=color_palette,
        data=df_fractions_to_true_conf_melted,
    )
    ax.set_xlabel(f'e(s): not {filter_relation}')
    ax.set_ylabel(error_metric_label)

    ax.set_title("SAR: Avg " + error_metric_label +
                 f" for rules predicting {target_relation}, 2 groups based on {filter_relation}\n"
                 f" (Avg over {n_rules}rules & {n_random_trials} random selections)\n"
                 f"e({target_relation}(s,*) |  {filter_relation}(s,*)) = {scar_propensity_score}        "
                 f" e({target_relation}(s,*) |  ~{filter_relation}(s, *)) in {prop_score_others_str} "
                 )

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plt.axvline(x=scar_propensity_score, color='k', linestyle='--')
    plt.text(x=scar_propensity_score, y=cwa_vertical_position, s="SCAR", verticalalignment='center')

    plt.tight_layout()
    plt.savefig(filename_image, bbox_inches='tight')
    plt.close(fig)


def generate_images_conf_comparison_known_prop_scores_sar_two_groups(
        dataset_name: str,
        target_relation: str,
        filter_relation: str,
        is_pca_version: bool,
        scar_propensity_score: float

):
    # dataset_name: str = "yago3_10"
    # target_relation: str = "haschild"
    # filter_relation: str = "ispoliticianof"
    # propensity_score_subjects_of_filter_relation: float = 0.8
    # propensity_score_other_entity: float = 0.2
    # # propensity_score_other_entities_list: List[float] = [0.2, 0.4, 0.6, 0.8, 1]
    # random_seed: int = 3
    # fraction_lower_bound: float = 0.1
    # fraction_upper_bound: float = 0.9
    # intersection_absolute_lower_bound: int = 10
    #
    # is_pca_version: bool = False

    if is_pca_version:
        pca_indicator: str = "pca_version"
        true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O
        conf_estimators_to_ignore: Set[ConfidenceEnum] = set()
    else:
        pca_indicator: str = "not_pca"
        true_conf: ConfidenceEnum = ConfidenceEnum.TRUE_CONF
        conf_estimators_to_ignore: Set[ConfidenceEnum] = set()

    root_dir_experiment_settings: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_experiment_high_level(
        dataset_name=dataset_name,
        target_relation=target_relation,
        filter_relation=filter_relation,
        is_pca_version=is_pca_version
    )

    image_dir: str = KnownPropScoresSARTwoGroupsFileNamer.get_dir_images(
        use_pca=is_pca_version, dataset_name=dataset_name, scar_propensity_score=scar_propensity_score
    )
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    df_rule_wrappers: pd.DataFrame = _get_rule_wrappers_as_dataframe_known_prop_scores_sar_two_groups(
        root_dir_experiment_settings=root_dir_experiment_settings,
        target_relation=target_relation,
        filter_relation=filter_relation,
        filter_group_prop_score=scar_propensity_score
    )

    column_names_logistics: List[str] = [
        'target_relation',
        'filter_relation',
        'prop_score_subj',
        'prop_score_other',
        'random_trial_index',
        'Rule',
    ]

    column_names_info = ColumnNamesInfo.build(
        true_conf=true_conf,
        conf_estimators_to_ignore=conf_estimators_to_ignore,
        column_names_logistics=column_names_logistics
    )

    df_rule_wrappers_relative_to_true_conf = df_rule_wrappers[
        column_names_info.get_df_rule_wrappers_columns_to_use()
    ]

    ########################################################################
    # Calc differences against true conf
    # df_diffs_true_conf_and_estimators: pd.DataFrame = get_df_rulewise_diffs_between_true_conf_and_conf_estimator(
    #     df_rule_wrappers=df_rule_wrappers,
    #     column_names_info=column_names_info
    # )
    #
    # df_diff_to_true_conf_melted: pd.DataFrame = df_diffs_true_conf_and_estimators.melt(
    #     id_vars=column_names_info.column_names_logistics,
    #     var_name='error_type',
    #     value_name='error_value'
    # )
    # df_diff_to_true_conf_melted['squared_error'] = df_diff_to_true_conf_melted.apply(
    #     lambda row: row['error_value'] ** 2,
    #     axis=1,
    #     result_type='reduce'
    # )
    df_diff_to_true_conf_melted: pd.DataFrame = get_df_diffs_between_true_conf_and_confidence_estimators_melted(
        df_rule_wrappers=df_rule_wrappers_relative_to_true_conf,
        column_names_info=column_names_info
    )

    n_rules = len(df_rule_wrappers_relative_to_true_conf["Rule"].unique())
    n_random_trials = len(df_rule_wrappers_relative_to_true_conf["random_trial_index"].unique())
    prop_score_other_list: List[float] = list(sorted(df_rule_wrappers_relative_to_true_conf["prop_score_other"].unique()))

    filename_root: str = f"known_prop_scores_sar" \
                         f"_{target_relation}" \
                         f"_{filter_relation}" \
                         f"_{pca_indicator}"

    ########################################################################
    if true_conf == ConfidenceEnum.TRUE_CONF:
        known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_vs_std_and_inverse_e_per_rule(
            df_rule_wrappers=df_rule_wrappers,
            filter_relation=filter_relation,
            scar_propensity_score=scar_propensity_score,
            image_dir=image_dir,
            filename_root=filename_root
        )
        known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule(
            df_rule_wrappers=df_rule_wrappers,
            filter_relation=filter_relation,
            scar_propensity_score=scar_propensity_score,
            image_dir=image_dir,
            filename_root=filename_root
        )
    elif true_conf == ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O or true_conf == ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S:
        pca_selection_mechanism_known_prop_scores_sar_two_subject_groups_plot_conf_evolution_true_conf_star_vs_pca_estimators_per_rule(
            df_rule_wrappers=df_rule_wrappers,
            filter_relation=filter_relation,
            image_dir=image_dir,
            filename_root=filename_root,
            scar_propensity_score=scar_propensity_score
        )

    ########################################################################

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

    generate_image_confidence_difference(
        true_conf=true_conf,
        df_diff_to_true_conf_melted=df_diff_to_true_conf_melted,
        target_relation=target_relation,
        filter_relation=filter_relation,
        n_rules=n_rules,
        n_random_trials=n_random_trials,
        column_names_info=column_names_info,
        scar_propensity_score=scar_propensity_score,
        prop_score_other_list=prop_score_other_list,
        filename_image=filename_image_confidence_difference,
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
        filter_relation=filter_relation,
        n_rules=n_rules,
        n_random_trials=n_random_trials,
        column_names_info=column_names_info,
        scar_propensity_score=scar_propensity_score,
        prop_score_other_list=prop_score_other_list,
        filename_image=filename_image_confidence_fraction
    )

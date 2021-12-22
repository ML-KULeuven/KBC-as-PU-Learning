from typing import List, Union, NamedTuple, Set, Dict

import pandas as pd

from kbc_pul.confidence_naming import ConfidenceEnum


class ColumnNamesInfo(NamedTuple):
    true_conf: ConfidenceEnum
    column_name_true_conf: str

    conf_estimators: List[ConfidenceEnum]
    column_names_conf_estimators: List[str]

    column_names_logistics: List[str]

    @staticmethod
    def build(true_conf: ConfidenceEnum, conf_estimators_to_ignore: Set[ConfidenceEnum],
              column_names_logistics: List[str]
              ) -> "ColumnNamesInfo":
        column_name_true_conf = true_conf.get_name()

        conf_estimators: List[ConfidenceEnum] = [
            conf
            for conf in ConfidenceEnum.get_estimators_of(true_conf)
            if conf not in conf_estimators_to_ignore
        ]

        column_names_conf_estimators: List[str] = [
            conf.get_name()
            for conf in conf_estimators
        ]
        return ColumnNamesInfo(
            true_conf=true_conf,
            column_name_true_conf=column_name_true_conf,
            conf_estimators=conf_estimators,
            column_names_conf_estimators=column_names_conf_estimators,
            column_names_logistics=column_names_logistics
        )

    def get_df_rule_wrappers_columns_to_use(self):
        return self.column_names_logistics + [self.column_name_true_conf] + self.column_names_conf_estimators

    def get_conf_estimators_color_palette(self) -> Dict[str, str]:
        color_palette: Dict[str, str] = {
            conf.get_name(): conf.get_hex_color_str()
            for conf in self.conf_estimators
        }
        return color_palette


def get_df_rulewise_diffs_between_true_conf_and_conf_estimator(
        df_rule_wrappers: pd.DataFrame,
        column_names_info: ColumnNamesInfo
) -> pd.DataFrame:
    df_rulewise_diffs_between_true_conf_and_conf_estimator: pd.DataFrame = df_rule_wrappers[
        column_names_info.column_names_logistics
    ]

    col_name_estimator: str
    for col_name_estimator in column_names_info.column_names_conf_estimators:
        df_rulewise_diffs_between_true_conf_and_conf_estimator \
            = df_rulewise_diffs_between_true_conf_and_conf_estimator.assign(
                **{
                    col_name_estimator: (
                            df_rule_wrappers[column_names_info.column_name_true_conf]
                            - df_rule_wrappers[col_name_estimator]
                    )
                }
                # col_name_estimator=
            )
        # df_rulewise_diffs_between_true_conf_and_conf_estimator[col_name_estimator] = (
        #         df_rule_wrappers[column_names_info.column_name_true_conf]
        #         - df_rule_wrappers[col_name_estimator]
        #
        #
        # )
    return df_rulewise_diffs_between_true_conf_and_conf_estimator


def get_df_rulewise_fraction_of_conf_estimator_to_true_conf(
        df_rule_wrappers: pd.DataFrame,
        column_names_info: ColumnNamesInfo
) -> pd.DataFrame:
    df_rulewise_fraction_of_conf_estimator_to_true_conf: pd.DataFrame = df_rule_wrappers[
        column_names_info.column_names_logistics
    ]

    col_name_estimator: str
    for col_name_estimator in column_names_info.column_names_conf_estimators:
        df_rulewise_fraction_of_conf_estimator_to_true_conf \
            = df_rulewise_fraction_of_conf_estimator_to_true_conf.assign(
                **{
                    col_name_estimator: (
                        df_rule_wrappers[col_name_estimator]
                        / df_rule_wrappers[column_names_info.column_name_true_conf]
                    )
                }
            )

    return df_rulewise_fraction_of_conf_estimator_to_true_conf


def get_df_rulewise_abs_and_squared_diffs_between_true_conf_and_conf_estimator(
        df_rule_wrappers: pd.DataFrame,
        column_names_logistics: List[str],
        column_name_true_conf: str,
        column_names_conf_estimators: List[str],

) -> pd.DataFrame:
    df_rulewise_abs_and_squared_diffs_between_true_conf_and_conf_estimator: pd.DataFrame = df_rule_wrappers[
        column_names_logistics
    ]

    col_name_estimator: str
    for col_name_estimator in column_names_conf_estimators:
        col_name_diff = f"Δ {col_name_estimator}"
        series_diff: pd.Series = (
                df_rule_wrappers[column_name_true_conf] - df_rule_wrappers[col_name_estimator]
        )
        df_rulewise_abs_and_squared_diffs_between_true_conf_and_conf_estimator[
            f"abs({col_name_diff})"
        ] = series_diff.abs()
        df_rulewise_abs_and_squared_diffs_between_true_conf_and_conf_estimator[
            f"abs({col_name_diff})"
        ] = series_diff.apply(lambda value: value ** 2)

    return df_rulewise_abs_and_squared_diffs_between_true_conf_and_conf_estimator


def get_df_diffs_between_true_conf_and_confidence_estimators_melted(
        df_rule_wrappers: pd.DataFrame,
        column_names_info: ColumnNamesInfo
) -> pd.DataFrame:
    df_diffs_true_conf_and_estimators: pd.DataFrame = get_df_rulewise_diffs_between_true_conf_and_conf_estimator(
        df_rule_wrappers=df_rule_wrappers,
        column_names_info=column_names_info
    )
    df_diffs_true_conf_and_estimators_melted: pd.DataFrame = df_diffs_true_conf_and_estimators.melt(
        id_vars=column_names_info.column_names_logistics,
        var_name='error_type',
        value_name='error_value'
    )
    df_diffs_true_conf_and_estimators_melted['error_value_abs'] = df_diffs_true_conf_and_estimators_melted[
        'error_value'
    ].abs()

    df_diffs_true_conf_and_estimators_melted['squared_error'] = df_diffs_true_conf_and_estimators_melted.apply(
        lambda row: row['error_value'] ** 2,
        axis=1,
        result_type='reduce'
    )

    return df_diffs_true_conf_and_estimators_melted

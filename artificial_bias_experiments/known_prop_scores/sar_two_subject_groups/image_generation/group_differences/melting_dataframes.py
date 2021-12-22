from typing import List, Callable, NamedTuple

import pandas as pd

from artificial_bias_experiments.known_prop_scores.sar_two_subject_groups.image_generation.group_differences.column_names import \
    CNameEnum, absolute_confidence_column_to_pretty_name, relative_conf_column_to_pretty_name, \
    absolute_n_predictions_column_to_pretty_name, relative_n_predictions_column_to_pretty_name, \
    absolute_pair_positive_conf_s_to_o_column_to_pretty_name


class MeltedDataFrameInfo(NamedTuple):
    df: pd.DataFrame
    cname_prediction_group: str
    cname_value_column: str


def get_melted_version(
        df_group_info: pd.DataFrame,
        non_id_columns: List[CNameEnum],
        var_name: str,
        value_name: str,
        function_transformation_to_pretty_group_names: Callable
) -> pd.DataFrame:
    # melt the DF
    df_melted = df_group_info.melt(
        id_vars=[
            str(col)
            for col in df_group_info.columns
            if col not in map(
                lambda x: x.value, non_id_columns
            )
        ],
        var_name=var_name,
        value_name=value_name
    )
    # Add column with pretty group names
    df_melted["prediction_group"] = df_melted.apply(
        lambda row: function_transformation_to_pretty_group_names(row[var_name]),
        axis=1
    )
    return df_melted

##################################################

# df_absolute_conf: pd.DataFrame = get_melted_version(
#     df_group_info=df_group_info,
#     non_id_columns=CNameEnum.get_absolute_confidence_cnames(),
#     var_name='group_type',
#     value_name='absolute_confidence_value',
#     function_transformation_to_pretty_group_names=absolute_confidence_column_to_pretty_name
# )


def get_df_abs_conf_per_group(df_group_info: pd.DataFrame) -> MeltedDataFrameInfo:

    cname_ugly_group_var_column: str = 'group_type'
    cname_prediction_group: str = 'prediction_group'
    cname_value_column: str = 'absolute_confidence_value'

    df_absolute_conf: pd.DataFrame = df_group_info.melt(
            id_vars=[
                str(col)
                for col in df_group_info.columns
                if col not in map(
                    lambda x: x.value, CNameEnum.get_absolute_confidence_cnames()
                )
            ],
            var_name=cname_ugly_group_var_column,
            value_name=cname_value_column
    )

    df_absolute_conf[cname_prediction_group] = df_absolute_conf.apply(
        lambda row: absolute_confidence_column_to_pretty_name(row[cname_ugly_group_var_column]),
        axis=1
    )
    return MeltedDataFrameInfo(
        df=df_absolute_conf,
        cname_prediction_group=cname_prediction_group,
        cname_value_column=cname_value_column
    )

###########################################
def get_df_rel_conf_per_group(df_group_info: pd.DataFrame) -> MeltedDataFrameInfo:

    cname_ugly_group_var_column: str = 'group_type'
    cname_prediction_group: str = 'prediction_group'
    cname_value_column: str = 'relative_confidence_value'

    df_relative_conf: pd.DataFrame = df_group_info.melt(
        id_vars=[
            str(col)
            for col in df_group_info.columns
            if col not in map(lambda x: x.value, CNameEnum.get_relative_confidence_cnames())
        ],
        var_name=cname_ugly_group_var_column,
        value_name=cname_value_column
    )

    df_relative_conf[cname_prediction_group] = df_relative_conf.apply(
        lambda row: relative_conf_column_to_pretty_name(row[cname_ugly_group_var_column]),
        axis=1
    )
    return MeltedDataFrameInfo(
        df=df_relative_conf,
        cname_prediction_group=cname_prediction_group,
        cname_value_column=cname_value_column
    )

############################################

def get_df_abs_pair_pos_conf_s_to_o_per_group(df_group_info: pd.DataFrame) -> MeltedDataFrameInfo:

    cname_ugly_group_var_column: str = 'group_type'
    cname_prediction_group: str = 'prediction_group'
    cname_value_column: str = 'absolute_true_pair_positive_conf_s_to_o'

    df_absolute_pair_pos_conf_s_to_o: pd.DataFrame = get_melted_version(
        df_group_info=df_group_info,
        non_id_columns=[
            CNameEnum.cname_true_pos_pair_conf_s_to_o,
            CNameEnum.cname_true_pos_pair_conf_s_to_o_in_filter,
            CNameEnum.cname_true_pos_pair_conf_s_to_o_not_in_filter,
        ],
        var_name=cname_ugly_group_var_column,
        value_name=cname_value_column,
        function_transformation_to_pretty_group_names=absolute_pair_positive_conf_s_to_o_column_to_pretty_name
    )
    return MeltedDataFrameInfo(
        df=df_absolute_pair_pos_conf_s_to_o,
        cname_prediction_group=cname_prediction_group,
        cname_value_column=cname_value_column
    )

####################################


def get_df_abs_n_predictions_per_group(df_group_info: pd.DataFrame) -> MeltedDataFrameInfo:

    cname_ugly_group_var_column: str = 'group_type'
    cname_prediction_group: str = 'prediction_group'
    cname_value_column: str = 'absolute_n_predictions_value'

    df_absolute_n_predictions: pd.DataFrame = df_group_info.melt(
        id_vars=[
            str(col)
            for col in df_group_info.columns
            if col not in map(lambda x: x.value, CNameEnum.get_absolute_n_prediction_cnames())
        ],
        var_name=cname_ugly_group_var_column,
        value_name=cname_value_column
    )

    df_absolute_n_predictions[cname_prediction_group] = df_absolute_n_predictions.apply(
        lambda row: absolute_n_predictions_column_to_pretty_name(row[cname_ugly_group_var_column]),
        axis=1
    )
    return MeltedDataFrameInfo(
        df=df_absolute_n_predictions,
        cname_prediction_group=cname_prediction_group,
        cname_value_column=cname_value_column
    )


def get_df_rel_n_predictions_per_group(df_group_info: pd.DataFrame) -> MeltedDataFrameInfo:

    cname_ugly_group_var_column: str = 'group_type'
    cname_prediction_group: str = 'prediction_group'
    cname_value_column: str = 'relative_n_predictions_value'

    df_relative_n_predictions: pd.DataFrame = df_group_info.melt(
        id_vars=[
            str(col)
            for col in df_group_info.columns
            if col not in map(lambda x: x.value, CNameEnum.get_relative_n_prediction_cnames())
        ],
        var_name=cname_ugly_group_var_column,
        value_name=cname_value_column
    )

    df_relative_n_predictions[cname_prediction_group] = df_relative_n_predictions.apply(
        lambda row: relative_n_predictions_column_to_pretty_name(row[cname_ugly_group_var_column]),
        axis=1
    )
    return MeltedDataFrameInfo(
        df=df_relative_n_predictions,
        cname_prediction_group=cname_prediction_group,
        cname_value_column=cname_value_column
    )



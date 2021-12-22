from enum import Enum
from typing import List, Dict

from kbc_pul.experiments_utils.color_utils import matplotlib_color_name_to_hex


class GroupNameEnum(Enum):
    total = "$\left|\mathbf{R}\\right|$"
    filter = '$S_{q}$'
    other = '$S_{\\neg q}$'

    @staticmethod
    def get_groups_as_ordered_strings() -> List[str]:
        return [
            GroupNameEnum.total.value,
            GroupNameEnum.filter.value,
            GroupNameEnum.other.value
        ]

    @ staticmethod
    def get_color_palette() -> Dict[str, str]:
        return {
            GroupNameEnum.total.value: matplotlib_color_name_to_hex("blue"),
            GroupNameEnum.filter.value: matplotlib_color_name_to_hex("green"),
            GroupNameEnum.other.value: matplotlib_color_name_to_hex("red")
        }


class CNameEnum(Enum):
    cname_true_conf = 'true_conf'
    cname_true_conf_in_filter = 'true_conf_on_predictions_in_filter'
    cname_true_conf_not_in_filter = 'true_conf_on_predictions_not_in_filter'

    cname_true_pos_pair_conf_s_to_o = 'true_pos_pair_conf_s_to_o'
    cname_true_pos_pair_conf_s_to_o_in_filter = 'true_pos_pair_s_to_o_conf_on_predictions_in_filter'
    cname_true_pos_pair_conf_s_to_o_not_in_filter = 'true_pos_pair_s_to_o_conf_on_predictions_not_in_filter'

    cname_true_pos_pair_conf_o_to_s = 'true_pos_pair_conf_o_to_s'
    cname_true_pos_pair_conf_o_to_s_in_filter = 'true_pos_pair_o_to_s_conf_on_predictions_in_filter'
    cname_true_pos_pair_conf_o_to_s_not_in_filter = 'true_pos_pair_o_to_s_conf_on_predictions_not_in_filter'

    cname_rel_true_conf_in_filter = 'rel_conf_in_filter'
    cname_rel_true_conf_not_in_filter = 'rel_conf_not_in_filter'

    cname_n_preds = 'n_predictions'
    cname_n_preds_in_filter = 'n_predictions_in_filter'
    cname_n_preds_not_in_filter = 'n_predictions_not_in_filter'

    cname_rel_n_preds_in_filter = 'rel_n_predictions_in_filter'
    cname_rel_n_preds_not_in_filter = 'rel_n_predictions_not_in_filter'

    cname_percentage_preds_in_filter = 'percentage_n_predictions_in_filter'
    cname_percentage_preds_not_in_filter = 'percentage_n_predictions_not_in_filter'

    @staticmethod
    def get_absolute_confidence_cnames() -> List['CNameEnum']:
        return [
            CNameEnum.cname_true_conf,
            CNameEnum.cname_true_conf_in_filter,
            CNameEnum.cname_true_conf_not_in_filter
        ]

    @staticmethod
    def get_relative_confidence_cnames() -> List['CNameEnum']:
        return [
            CNameEnum.cname_rel_true_conf_in_filter,
            CNameEnum.cname_rel_true_conf_not_in_filter
        ]

    @staticmethod
    def get_absolute_n_prediction_cnames() -> List['CNameEnum']:
        return [
            CNameEnum.cname_n_preds,
            CNameEnum.cname_n_preds_in_filter,
            CNameEnum.cname_n_preds_not_in_filter
        ]

    @staticmethod
    def get_relative_n_prediction_cnames() -> List['CNameEnum']:
        return [
            CNameEnum.cname_rel_n_preds_in_filter,
            CNameEnum.cname_rel_n_preds_not_in_filter
        ]


def absolute_confidence_column_to_pretty_name(old_column_name: str) -> str:
    if old_column_name == CNameEnum.cname_true_conf.value:
        return GroupNameEnum.total.value
    elif old_column_name == CNameEnum.cname_true_conf_in_filter.value:
        return GroupNameEnum.filter.value
    elif old_column_name == CNameEnum.cname_true_conf_not_in_filter.value:
        return GroupNameEnum.other.value
    else:
        raise Exception()


def relative_conf_column_to_pretty_name(old_column_name: str) -> str:
    if old_column_name == CNameEnum.cname_rel_true_conf_in_filter.value:
        return GroupNameEnum.filter.value
    elif old_column_name == CNameEnum.cname_rel_true_conf_not_in_filter.value:
        return GroupNameEnum.other.value
    else:
        raise Exception()


def absolute_pair_positive_conf_s_to_o_column_to_pretty_name(old_column_name: str) -> str:
    if old_column_name == CNameEnum.cname_true_pos_pair_conf_s_to_o.value:
        return GroupNameEnum.total.value
    elif old_column_name == CNameEnum.cname_true_pos_pair_conf_s_to_o_in_filter.value:
        return GroupNameEnum.filter.value
    elif old_column_name == CNameEnum.cname_true_pos_pair_conf_s_to_o_not_in_filter.value:
        return GroupNameEnum.other.value
    else:
        raise Exception()

def absolute_n_predictions_column_to_pretty_name(old_column_name: str)-> str:
    if old_column_name == CNameEnum.cname_n_preds.value:
        return GroupNameEnum.total.value
    elif old_column_name == CNameEnum.cname_n_preds_in_filter.value:
        return GroupNameEnum.filter.value
    elif old_column_name == CNameEnum.cname_n_preds_not_in_filter.value:
        return GroupNameEnum.other.value
    else:
        raise Exception()


def relative_n_predictions_column_to_pretty_name(old_column_name: str) -> str:
    if old_column_name == CNameEnum.cname_rel_n_preds_in_filter.value:
        return GroupNameEnum.filter.value
    elif old_column_name == CNameEnum.cname_rel_n_preds_not_in_filter.value:
        return GroupNameEnum.other.value
    else:
        raise Exception()

from enum import Enum
from typing import List, Dict

from kbc_pul.experiments_utils.color_utils import matplotlib_color_name_to_hex


class ConfidenceEnum(Enum):
    TRUE_CONF = '$conf$'
    # E_CONF_EST = '$\widehat{conf}_{1/e}$'
    IPW_CONF = 'IPW'
    # STD_CONF_EST = '$\widehat{conf}_{std}$'
    CWA_CONF = 'CWA'
    # INVERSE_C_CONF_EST = '$\widehat{conf}_\\frac{1}{c}$'
    ICW_CONF = 'ICW'

    # TRUE_CONF_STAR_S_TO_O = '$conf^*$ (S->O)'
    # TRUE_CONF_STAR_S_TO_O = '$conf^*$(S)'
    TRUE_CONF_BIAS_YS_ZERO_S_TO_O = '$\\frac{\left| \mathbf{R}\\right|}{\left| \mathbf{R_s}\\right|} conf$ $p$'
    # PCA_CONF_STAR_EST_S_TO_O = '$\widehat{conf^*}_{PCA}$ (S->O)'
    PCA_CONF_S_TO_O = 'PCA $p$'
    # E_PCA_CONF_STAR_EST_S_TO_O = '$\widehat{conf^*}_{e+PCA}$ (S->O)'
    IPW_PCA_CONF_S_TO_O = 'IPW-PCA $p$'

    # TRUE_CONF_STAR_O_TO_S = '$conf^*$ (O->S)'
    # TRUE_CONF_STAR_O_TO_S = '$conf^*$(O)'
    TRUE_CONF_BIAS_YS_ZERO_O_TO_S = '$\\frac{\left| \mathbf{R}\\right|}{\left| \mathbf{R_s}\\right|} conf$ $p^{-1}$'
    # PCA_CONF_STAR_EST_O_TO_S = '$\widehat{conf^*}_{PCA}$ (O->S)'
    PCA_CONF_O_TO_S = 'PCA ${p^{-1}}$'
    # E_PCA_CONF_STAR_EST_O_TO_S = '$\widehat{conf^*}_{e+PCA}$ (O->S)'
    IPW_PCA_CONF_O_TO_S = 'IPW-PCA ${p^{-1}}$'

    def get_hex_color_str(self) -> str:
        return self._color_dict.get(self, matplotlib_color_name_to_hex("black"))

    @staticmethod
    def get_propensity_weighted_estimators() -> List['ConfidenceEnum']:
        return [
            ConfidenceEnum.IPW_CONF,
            ConfidenceEnum.IPW_PCA_CONF_S_TO_O,
            ConfidenceEnum.IPW_PCA_CONF_O_TO_S
        ]

    @staticmethod
    def get_true_confidences() -> List['ConfidenceEnum']:
        return [ConfidenceEnum.TRUE_CONF, ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O, ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S]

    @staticmethod
    def get_estimators_of(true_confidence: 'ConfidenceEnum'):
        if true_confidence is ConfidenceEnum.TRUE_CONF:
            return [
                ConfidenceEnum.IPW_CONF,
                ConfidenceEnum.CWA_CONF,
                ConfidenceEnum.ICW_CONF,
                ConfidenceEnum.PCA_CONF_S_TO_O,
                ConfidenceEnum.PCA_CONF_O_TO_S,
            ]
        elif true_confidence is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O:
            return [
                ConfidenceEnum.PCA_CONF_S_TO_O,
                ConfidenceEnum.IPW_PCA_CONF_S_TO_O
            ]
        elif true_confidence is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S:
            return [
                ConfidenceEnum.PCA_CONF_O_TO_S,
                ConfidenceEnum.IPW_PCA_CONF_O_TO_S
            ]
        else:
            raise Exception(f"Cannot return estimators of {true_confidence}")

    def get_name(self):
        return self.value

    # def get_color(self):


def get_color_dict_q1() -> Dict[ConfidenceEnum, str]:
    return {
        ConfidenceEnum.IPW_CONF: matplotlib_color_name_to_hex("red"),
        ConfidenceEnum.CWA_CONF: matplotlib_color_name_to_hex("purple"),
        ConfidenceEnum.ICW_CONF: matplotlib_color_name_to_hex("orange"),

        ConfidenceEnum.PCA_CONF_S_TO_O: matplotlib_color_name_to_hex("blue"),
        ConfidenceEnum.IPW_PCA_CONF_S_TO_O: matplotlib_color_name_to_hex("cyan"),

        ConfidenceEnum.PCA_CONF_O_TO_S: matplotlib_color_name_to_hex("green"),
        ConfidenceEnum.IPW_PCA_CONF_O_TO_S: matplotlib_color_name_to_hex("lime"),

        ConfidenceEnum.TRUE_CONF: matplotlib_color_name_to_hex("fuchsia"),
        ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O: matplotlib_color_name_to_hex("navy"),
        ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S: matplotlib_color_name_to_hex("darkgreen")
    }
"""
#1b9e77
#d95f02
#7570b3

#e7298a
#66a61e
#e6ab02
"""


ConfidenceEnum._color_dict = {

        ConfidenceEnum.TRUE_CONF: matplotlib_color_name_to_hex("black"),

        ConfidenceEnum.IPW_CONF: "#f781bf",
        ConfidenceEnum.CWA_CONF: "#377eb8",
        ConfidenceEnum.ICW_CONF: "#4daf4a",

        ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O: "#1b9e77",
        ConfidenceEnum.PCA_CONF_S_TO_O: "#d95f02",
        ConfidenceEnum.IPW_PCA_CONF_S_TO_O: "#7570b3",

        ConfidenceEnum.PCA_CONF_O_TO_S: "#e7298a",
        ConfidenceEnum.IPW_PCA_CONF_O_TO_S: "#66a61e",
        ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S: "#e6ab02",
    }



if __name__ == '__main__':
    for conf in ConfidenceEnum:
        print(conf.get_name())

    print("\nTrue confidences:")
    for conf in ConfidenceEnum.get_true_confidences():
        print(conf.value)

    print("\nEstimators of true conf")
    for conf in ConfidenceEnum.get_estimators_of(ConfidenceEnum.TRUE_CONF):
        print(conf.get_name())

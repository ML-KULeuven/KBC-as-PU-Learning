from kbc_pul.confidence_naming import ConfidenceEnum


def get_confidence_estimate_label_string(true_conf: ConfidenceEnum) -> str:
    if true_conf is ConfidenceEnum.TRUE_CONF:
        error_metric = "$\widehat{conf}(R)$"
    elif true_conf is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O or true_conf is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S:
        error_metric = "$\widehat{conf^*}(R)$"
    else:
        raise Exception(f"Confidence estimator label not defined for {true_conf}")
    return error_metric


def get_confidence_difference_label_string(true_conf: ConfidenceEnum) -> str:
    if true_conf is ConfidenceEnum.TRUE_CONF:
        error_metric = "$[conf(R) - \widehat{conf}(R)]^2$"
    elif true_conf is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O or true_conf is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S:
        error_metric = "$[conf^*(R) - \widehat{conf^*}(R)]^2$"
    else:
        raise Exception(f"Confidence difference label not defined for {true_conf}")
    return error_metric


def get_confidence_fraction_label_string(true_conf: ConfidenceEnum) -> str:
    if true_conf is ConfidenceEnum.TRUE_CONF:
        error_metric = "$\\frac{\widehat{conf}(R)}{conf(R)}$"
    elif true_conf is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O or true_conf is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S:
        error_metric = "$\\frac{\widehat{conf^*}(R)}{conf^*(R)}$"
    else:
        raise Exception(f"Confidence difference label not defined for {true_conf}")
    return error_metric

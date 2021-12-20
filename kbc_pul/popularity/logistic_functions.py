import math


def logistic_function(x,
                      x_zero: float = 0.0,
                      max_value: float = 1.0,
                      log_growth_rate: float = 1.0,
                      ) -> float:
    """


    :param x: the input point in the domain
    :param x_zero: the x-value of the sigmoid's midpoint
    :param max_value: the curve's maximum value
    :param log_growth_rate: the logistic growth rate or steepness of the curve


    :return:
    """

    exponent: float = - log_growth_rate * (x - x_zero)
    denominator = 1 + math.exp(exponent)

    value = max_value / denominator

    return value


def logistic_popularity_function(x: float, log_growth_rate: float = 1.0) -> float:
    return 2 * logistic_function(x, log_growth_rate=log_growth_rate) - 1

from random import Random


def make_random_observation_choice(random: Random, propensity_score: float) -> bool:
    """
    Decide whether a row should be observed or not
        - generate a uniform value between [0,1]
        - accept if the value is smaller than the propensity score

    :param random:
    :param propensity_score:
    :return:
    """
    random_value: float = random.uniform(0, 1)
    return random_value <= propensity_score

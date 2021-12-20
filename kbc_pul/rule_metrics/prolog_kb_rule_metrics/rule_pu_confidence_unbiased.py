from typing import Dict, Optional, Union

import pandas as pd
from pylo.language.lp import (Atom as PyloAtom)


class PropensityScoreCounter:

    def __init__(self):
        self.propensity_score_to_literal_count_dict: Dict[float, int] = dict()

    def increment_literal_count_for_score(self, propensity_score: float) -> None:
        if self.propensity_score_to_literal_count_dict.get(propensity_score, None) is None:
            self.propensity_score_to_literal_count_dict[propensity_score] = 1
        else:
            self.propensity_score_to_literal_count_dict[propensity_score] = self.propensity_score_to_literal_count_dict[
                                                                                propensity_score] + 1

    def to_str(self):
        dict_repr = "{\n\t" + ",\n\t ".join(
            [f"{val}: {key:0.2f} (inverse: {1.0 / key:0.2f}) ==> {val} exs -> {1.0 / key * val:0.2f} exs" for key, val
             in
             self.propensity_score_to_literal_count_dict.items()]) + "}"
        return dict_repr


def get_inverse_propensity_score_for_literal(
        predicted_literal: Union[PyloAtom, pd.Series],
        propensity_score_controller,
        propensity_score_literal_counter: Optional[PropensityScoreCounter] = None
) -> float:
    literal_propensity_score: float = propensity_score_controller.get_propensity_score_of(predicted_literal)
    if abs(literal_propensity_score) < 0.0001:
        # raise Exception("This should not occur")
        raise Exception(f"EXTREMELY LOW PROPENSITY SCORE {literal_propensity_score}")
        # print(f"SUPER LOW PROPENSITY SCORE {literal_propensity_score}")
    else:
        if propensity_score_literal_counter is not None:
            propensity_score_literal_counter.increment_literal_count_for_score(literal_propensity_score)
        inverse_propensity_score = 1.0 / literal_propensity_score
        return inverse_propensity_score

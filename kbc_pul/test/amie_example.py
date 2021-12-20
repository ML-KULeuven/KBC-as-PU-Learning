from typing import List

import pandas as pd

data: List[List[str]] = [
    ['adam', 'livesin', 'paris'],
    ['adam', 'livesin', 'rome'],
    ['bob', 'livesin', 'zurich'],
    ['adam', 'wasbornin', 'paris'],
    ['carl', 'wasbornin', 'rome']
]
columns = ["Subject", "Rel", "Object"]

rule_string = "bornin(X,Y) :- livesin(X,Y)"

expected_cwa_confidence: float = 1/3
expected_pca_confidence: float = 1/2

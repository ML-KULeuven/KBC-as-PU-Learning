from typing import NamedTuple, List, Optional


class PropScoresTwoSARGroups(NamedTuple):
    in_filter: float
    other: float

    @staticmethod
    def get_column_names(prefix: Optional = None) -> List[str]:
        if prefix is None:
            prefix = ""

        return [
            f"{prefix}prop_scores_in_filter",
            f"{prefix}prop_scores_not_in_filter",
        ]

    def to_rows(self) -> List[float]:
        return [
            self.in_filter,
            self.other
        ]

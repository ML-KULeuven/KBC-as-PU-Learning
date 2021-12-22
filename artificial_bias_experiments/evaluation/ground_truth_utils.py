from typing import NamedTuple, Set, Tuple

import pandas as pd


class TrueEntitySetsTuple(NamedTuple):
    entity_pairs: Set[Tuple[str, str]]
    pca_subjects: Set[str]
    pca_objects: Set


def get_true_entity_sets_as_string_sets(
    df_ground_truth_target_relation: pd.DataFrame
) -> TrueEntitySetsTuple:
    true_target_relation_ent_pairs: Set[Tuple[str, str]] = set(
        df_ground_truth_target_relation.apply(
            lambda row: (row["Subject"], row["Object"]),
            axis=1,
            result_type='reduce'
        )
    )
    true_pca_subject_set: Set[str] = set(df_ground_truth_target_relation["Subject"].unique())
    true_pca_object_set: Set[str] = set(df_ground_truth_target_relation["Object"].unique())
    return TrueEntitySetsTuple(true_target_relation_ent_pairs, true_pca_subject_set, true_pca_object_set)

from typing import List, Set, Iterable, Dict, Optional, Union, Sequence, Tuple

import numpy as np
import pandas as pd

from pylo.language.lp import (Clause as PyloClause, Atom as PyloAtom, Not as PyloNot, Body as PyloBody)


def check_that_triple_dataframe_has_sro_format(df_triples: pd.DataFrame) -> None:
    """
    Check that the given argument is a dataframe with exactly 3 columns: {"Subject", "Rel", "Object"}
    Throw Exception otherwise.

    :param df_triples:
    :return: None
    """
    n_columns: int = df_triples.shape[1]
    expected_columns: Set[str] = {"Subject", "Rel", "Object"}
    if n_columns != len(expected_columns):
        raise Exception(f"Expected {len(expected_columns)} columns {', '.join(expected_columns)}"
                        f" but got {n_columns} columns: {', '.join([str(col) for col in df_triples.columns])}")

    for col in df_triples.columns:
        if col not in expected_columns:
            raise Exception(
                f"Did not expect column named {col}; only expected columns {', '.join(expected_columns)}")


class PandasKnowledgeBaseWrapper:
    """
    Knowledge Base represented using pandas dataframes

    """
    column_subject_of_predictions: str = "Subject"
    column_object_of_predictions: str = "Object"

    @staticmethod
    def create_from_full_data(df_full_data: pd.DataFrame) -> 'PandasKnowledgeBaseWrapper':
        """
        Creates a KB represented using a dataframe of triples.
        The input dataframe has to have 3 columns: "Subject", "Rel", "Object", and every row represents a fact.

        The created PandasKnowledgeBaseWrapper maps every relation to a (Subject, Object)-Dataframe representing
        the facts for that relation in the KB

        :param df_full_data: input dataframe with 3 columns: "Subject", "Rel", "Object". Every row represents a fact.
        :return:
        """
        check_that_triple_dataframe_has_sro_format(df_triples=df_full_data)

        all_predicates: Iterable[str] = df_full_data["Rel"].unique()

        map_of_predicate_to_df_entity_tuples: Dict[str, pd.DataFrame] = dict()
        predicate: str
        for predicate in all_predicates:
            df_relation: pd.DataFrame = df_full_data[df_full_data["Rel"] == predicate][["Subject", "Object"]]
            map_of_predicate_to_df_entity_tuples[predicate] = df_relation
        return PandasKnowledgeBaseWrapper(map_of_predicate_to_df_entity_tuples=map_of_predicate_to_df_entity_tuples)

    def __init__(self, map_of_predicate_to_df_entity_tuples: Dict[str, pd.DataFrame]):
        """
        Representation of a Knowledge Base (KB), using a mapping. The mapping maps the name of every relation onto
        a Pandas DataFrame containing 2 columns (Subject & Object), representing the facts for that relation ih the KB.

        :param map_of_predicate_to_df_entity_tuples: maps the relation name onto a 2-column dataframe representing
        the facts having that relation.
        """
        self.map_of_predicate_to_df_entity_tuples: Dict[str, pd.DataFrame] = map_of_predicate_to_df_entity_tuples

        self.string_repr = "{\n " + ",\n".join(
            [f"\t{relation}: {df.shape[0]} lits"
             for relation, df
             in sorted(self.map_of_predicate_to_df_entity_tuples.items())
             ]
        ) + "\n}"

    def get_relation(self, relation: str) -> pd.DataFrame:
        return self.map_of_predicate_to_df_entity_tuples[relation]

    def replace_predicate(self, relation: str, new_df_for_relation: pd.DataFrame) -> pd.DataFrame:
        old_df: pd.DataFrame = self.map_of_predicate_to_df_entity_tuples[relation]
        self.map_of_predicate_to_df_entity_tuples[relation] = new_df_for_relation
        return old_df

    def __str__(self):
        return self.string_repr

    def __repr__(self):
        return self.__str__()

    def get_predictions_for_rule(self, rule: PyloClause) -> Optional[pd.DataFrame]:
        rule_head_variable_names: List[str] = [
            variable.get_name()
            for variable in rule.get_head().get_variables()
        ]
        # print(rule_head_variable_names)
        list_rule_body_literals: Sequence[Union[PyloAtom, PyloNot]] = rule.get_body().get_literals()

        current_body_predictions: Optional[pd.DataFrame] = None
        body_lit: PyloAtom
        for body_lit in list_rule_body_literals:
            rel_of_body_lit: str = body_lit.predicate.get_name()
            df_rel_ents: pd.DataFrame = self.map_of_predicate_to_df_entity_tuples[rel_of_body_lit]
            body_lit_variable_names: List[str] = [arg.get_name() for arg in body_lit.get_arguments()]

            renaming_dict: Dict = {
                "Subject": body_lit_variable_names[0],
                "Object": body_lit_variable_names[1]
            }

            if current_body_predictions is None:
                current_body_predictions = df_rel_ents.rename(
                    columns=renaming_dict
                )
            else:
                # obtain new current predictions by inner join between
                #   current predictions
                #   df of current predictions

                # get the intersection of variables of body up until now and the current literal
                current_variables: List[str] = current_body_predictions.columns
                # Join columns with variables as column names
                join_columns_current_body_predictions: List[str] = [
                    var_of_literal
                    for var_of_literal in body_lit_variable_names
                    if var_of_literal in current_variables
                ]
                if len(join_columns_current_body_predictions) > 2:
                    raise Exception(f"Unexpected number of join columns: {join_columns_current_body_predictions}")
                if len(join_columns_current_body_predictions) == 0:
                    raise Exception(f"No shared columns:"
                                    f" literal {body_lit} is not connected to the earlier part of the rule body")

                # Join columns of literal with Subject, Object as column names
                join_columns_df_rel_ents: List[str] = []
                for entity_column, variable_name in renaming_dict.items():
                    if variable_name in join_columns_current_body_predictions:
                        join_columns_df_rel_ents.append(entity_column)

                # Do the join
                tmp = current_body_predictions.merge(
                    df_rel_ents,
                    left_on=join_columns_current_body_predictions,
                    right_on=join_columns_df_rel_ents
                ).drop(join_columns_df_rel_ents, axis=1).rename(columns=renaming_dict)
                current_body_predictions = tmp

            if len(current_body_predictions) == 0:
                return None

        current_body_predictions = current_body_predictions[rule_head_variable_names].drop_duplicates().rename(
            columns={
                rule_head_variable_names[0]: "Subject",
                rule_head_variable_names[1]: "Object"
            }
        )[["Subject", "Object"]].reset_index(drop=True)

        return current_body_predictions

    def does_example_satisfy_rule_body(self,
                                       body: PyloBody,
                                       # rule: PyloClause,
                                       head_variable_name: str,
                                       head_value_name: str
                                       ) -> Optional[pd.DataFrame]:

        if not isinstance(head_value_name, str):
            head_value_name = str(head_value_name)

        list_rule_body_literals: Sequence[Union[PyloAtom, PyloNot]] = body.get_literals()
        head_literal = list_rule_body_literals[0]
        list_rule_body_literals = list_rule_body_literals[1:]
        rule_head_variable_names: List[str] = [
            variable.get_name()
            for variable in head_literal.get_variables()
        ]
        # print(rule_head_variable_names)
        # list_rule_body_literals: Sequence[Union[PyloAtom, PyloNot]] = rule.get_body().get_literals()
        # list_rule_body_literals: Sequence[Union[PyloAtom, PyloNot]] = body.get_literals()

        current_body_predictions: Optional[pd.DataFrame] = None
        body_lit: PyloAtom
        for body_lit in list_rule_body_literals:
            rel_of_body_lit: str = body_lit.predicate.get_name()
            df_rel_ents: pd.DataFrame = self.map_of_predicate_to_df_entity_tuples[rel_of_body_lit]
            body_lit_variable_names: List[str] = [arg.get_name() for arg in body_lit.get_arguments()]

            renaming_dict: Dict = {
                "Subject": body_lit_variable_names[0],
                "Object": body_lit_variable_names[1]
            }

            if current_body_predictions is None:
                current_body_predictions = df_rel_ents.rename(
                    columns=renaming_dict
                )
            else:
                # obtain new current predictions by inner join between
                #   current predictions
                #   df of current predictions

                # get the intersection of variables of body up until now and the current literal
                current_variables: List[str] = current_body_predictions.columns
                # Join columns with variables as column names
                join_columns_current_body_predictions: List[str] = [
                    var_of_literal
                    for var_of_literal in body_lit_variable_names
                    if var_of_literal in current_variables
                ]
                if len(join_columns_current_body_predictions) > 2:
                    raise Exception(f"Unexpected number of join columns: {join_columns_current_body_predictions}")
                if len(join_columns_current_body_predictions) == 0:
                    raise Exception(f"No shared columns:"
                                    f" literal {body_lit} is not connected to the earlier part of the rule body")

                # Join columns of literal with Subject, Object as column names
                join_columns_df_rel_ents: List[str] = []
                for entity_column, variable_name in renaming_dict.items():
                    if variable_name in join_columns_current_body_predictions:
                        join_columns_df_rel_ents.append(entity_column)

                # Do the join
                tmp = current_body_predictions.merge(
                    df_rel_ents,
                    left_on=join_columns_current_body_predictions,
                    right_on=join_columns_df_rel_ents
                ).drop(join_columns_df_rel_ents, axis=1).rename(columns=renaming_dict)
                current_body_predictions = tmp

            for body_lit_variable_name in body_lit_variable_names:
                if body_lit_variable_name == head_variable_name:
                    current_body_predictions = current_body_predictions[
                        current_body_predictions[body_lit_variable_name] == head_value_name]

            if len(current_body_predictions) == 0:
                return None

        # current_body_predictions = current_body_predictions[rule_head_variable_names].drop_duplicates().rename(
        #     columns={
        #         rule_head_variable_names[0]: "Subject",
        #         rule_head_variable_names[1]: "Object"
        #     }
        # )[["Subject", "Object"]].reset_index(drop=True)

        return current_body_predictions

    def get_pca_status_for_predictions(self, df_predictions: pd.DataFrame, target_relation: str) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        self.check_if_df_in_pandas_kb_format(df_entity_tuples=df_predictions)

        df_target_relation: pd.DataFrame = self.map_of_predicate_to_df_entity_tuples[target_relation]

        e1_entity_set: np.ndarray = df_target_relation["Subject"].unique()
        e2_entity_set: np.ndarray = df_target_relation["Object"].unique()

        exists_lits_same_subject: np.ndarray = np.isin(
            df_predictions[self.column_subject_of_predictions].values, e1_entity_set
        )
        exists_lits_same_object: np.ndarray = np.isin(
            df_predictions[self.column_object_of_predictions].values, e2_entity_set
        )
        return exists_lits_same_subject, exists_lits_same_object

    def are_predictions_supported(self, df_predictions: pd.DataFrame, target_relation: str) -> pd.Series:
        self.check_if_df_in_pandas_kb_format(df_entity_tuples=df_predictions)
        df_target_relation: pd.DataFrame = self.map_of_predicate_to_df_entity_tuples[target_relation]

        is_supported: pd.Series = df_predictions.apply(
            lambda row: (
                np.any(
                    (df_target_relation["Subject"] == row[self.column_subject_of_predictions])
                    & (df_target_relation["Object"] == row[self.column_object_of_predictions])
                )
            ),
            axis=1,
            result_type='reduce'
        )
        return is_supported

    @staticmethod
    def check_if_df_in_pandas_kb_format(df_entity_tuples: pd.DataFrame):
        if df_entity_tuples is None or len(df_entity_tuples) == 0:
            raise Exception("Requires non-empty prediction DF")

        expected_columns: Set[str] = {
            PandasKnowledgeBaseWrapper.column_subject_of_predictions,
            PandasKnowledgeBaseWrapper.column_object_of_predictions
        }
        current_columns: Set[str] = set(df_entity_tuples.columns)
        if not expected_columns <= current_columns:
            raise Exception(f"Expected columns {', '.join(expected_columns)}"
                            f" but got columns: {', '.join([str(col) for col in df_entity_tuples.columns])}")

    def calculate_prediction_cache_for_rule(self, rule: PyloClause) -> Optional[pd.DataFrame]:
        o_df_predictions: Optional[pd.DataFrame] = self.get_predictions_for_rule(rule=rule)
        if o_df_predictions is None:
            return None
        else:
            target_relation: str = rule.get_head().get_predicate().get_name()

            o_df_predictions["is_supported"] = self.are_predictions_supported(
                df_predictions=o_df_predictions,
                target_relation=target_relation,
            )
            exists_lits_same_subject: np.ndarray
            exists_lits_same_object: np.ndarray
            exists_lits_same_subject, exists_lits_same_object = self.get_pca_status_for_predictions(
                df_predictions=o_df_predictions,
                target_relation=target_relation
            )
            o_df_predictions['exists_lits_same_subject'] = exists_lits_same_subject
            o_df_predictions['exists_lits_same_object'] = exists_lits_same_object
            return o_df_predictions

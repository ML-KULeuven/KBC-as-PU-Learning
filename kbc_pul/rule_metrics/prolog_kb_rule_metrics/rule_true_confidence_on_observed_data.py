from typing import Set, Tuple, Optional

from pylo.language.lp import (Constant as PyloConstant, Context as PyloContext)

from kbc_pul.data_structures.rule_wrapper import RuleWrapper


def set_of_string_tuples_to_set_of_pylo_const_tuples(
        set_of_entity_string_tuples: Set[Tuple[str, str]],
        pylo_context: PyloContext
) -> Set[Tuple[PyloConstant, PyloConstant]]:
    set_of_entity_pylo_constant_tuples: Set[Tuple[PyloConstant, PyloConstant]] = set()
    entity_string_tuple: Tuple[str, str]
    for entity_string_tuple in set_of_entity_string_tuples:
        set_of_entity_pylo_constant_tuples.add(
            (
                pylo_context.constant(entity_string_tuple[0]),
                pylo_context.constant(entity_string_tuple[1])
            )
        )
    return set_of_entity_pylo_constant_tuples


def set_of_pylo_const_tuples_to_set_of_string_tuples(
        set_of_pylo_const_tuples: Set[Tuple[PyloConstant, PyloConstant]],
) -> Set[Tuple[str, str]]:
    set_of_entity_string_tuples: Set[Tuple[str, str]] = {
        (pylo_cst_tuple[0].name, pylo_cst_tuple[1].name)
        for pylo_cst_tuple in set_of_pylo_const_tuples
    }
    return set_of_entity_string_tuples


def get_true_confidence_on_observed_data(
        predictions_as_pylo_const_tuple_set: Set[Tuple[PyloConstant, PyloConstant]],
        true_entity_tuple_set: Set[Tuple[PyloConstant, PyloConstant]],
        verbose: bool = False
) -> float:
    n_predictions: int = len(predictions_as_pylo_const_tuple_set)

    if len(predictions_as_pylo_const_tuple_set) != 0 and len(true_entity_tuple_set) != 0:
        single_prediction = next(iter(predictions_as_pylo_const_tuple_set))
        single_true_tuple = next(iter(true_entity_tuple_set))
        first_el_single_prediction = single_prediction[0]
        first_el_true_tuple = single_true_tuple[0]
        if type(first_el_true_tuple) != type(first_el_single_prediction):
            raise Exception(f"Two different types: {type(first_el_true_tuple)} {type(first_el_single_prediction)}")

    true_prediction_set: Set[Tuple[PyloConstant, PyloConstant]] = predictions_as_pylo_const_tuple_set.intersection(
        true_entity_tuple_set)
    n_true_predictions: int = len(true_prediction_set)
    if verbose:
        print(f"{n_true_predictions} / {n_predictions} true predictions")
    return n_true_predictions / n_predictions


def get_pca_confidence_on_observed_data(
        predictions_as_pylo_const_tuple_set: Set[Tuple[PyloConstant, PyloConstant]],
        true_entity_tuple_set: Set[Tuple[PyloConstant, PyloConstant]],
        pca_non_target_entity_set: Set[str],
        predict_object_entity: bool,
        verbose: bool = False
) -> Optional[float]:
    """
    Given a rule r: h(X,Y) <= B,

    """
    if predict_object_entity:
        non_target_index = 0
    else:
        non_target_index = 1

    n_known_positive_predictions: int = 0
    n_pca_negative_predictions: int = 0
    n_pca_unknown_predictions: int = 0

    predicted_entity_tuple: Tuple[PyloConstant, PyloConstant]
    for predicted_entity_tuple in predictions_as_pylo_const_tuple_set:
        does_literal_occur_in_kb: bool = predicted_entity_tuple in true_entity_tuple_set
        if does_literal_occur_in_kb:
            # if verbose:
            #     print(f"\t{predicted_entity_tuple} occurs in KB")
            n_known_positive_predictions += 1
        else:
            non_target_entity_str: str = predicted_entity_tuple[non_target_index].name
            exist_other_outgoing_edges: bool = non_target_entity_str in pca_non_target_entity_set
            if exist_other_outgoing_edges:
                # if verbose:
                #     print(f"\t{predicted_entity_tuple} does NOT in KB, {non_target_entity_str} IS IN set -> PCA negative")

                n_pca_negative_predictions += 1
            else:
                # if verbose:
                #     print(f"\t{predicted_entity_tuple} does NOT in KB, {non_target_entity_str} IS IN set -> PCA UNKNOWN")
                n_pca_unknown_predictions += 1
        # if verbose:
        #     print()

    if verbose:
        print(f"target : {'object' if predict_object_entity else 'subject'}")
        print(f"# KPs : {n_known_positive_predictions}")
        print(f"# pca negatives : {n_pca_negative_predictions}")

    pca_body_support: int = n_known_positive_predictions + n_pca_negative_predictions
    if pca_body_support == 0:
        print(f"PCA body support is 0 (for {len(predictions_as_pylo_const_tuple_set)} predictions)")
        print("--> cannot compute PCA confidence on observed data")
        return None
    else:
        pca_confidence: float = n_known_positive_predictions / pca_body_support
        return pca_confidence


EntityStr = str


def calculate_true_confidence_metrics(
        rule_wrapper: RuleWrapper,
        predictions_as_pylo_const_tuple_set: Set[Tuple[PyloConstant, PyloConstant]],
        true_entity_tuple_set: Set[Tuple[PyloConstant, PyloConstant]],
        pca_subject_set: Set[EntityStr],
        pca_object_set: Set[EntityStr],
        verbose: bool = False
) -> None:
    if len(predictions_as_pylo_const_tuple_set) != 0:

        if verbose:
            print("\tcalculating true confidences")

        true_conf: float = get_true_confidence_on_observed_data(
            predictions_as_pylo_const_tuple_set=predictions_as_pylo_const_tuple_set,
            true_entity_tuple_set=true_entity_tuple_set,
            verbose=verbose
        )
        rule_wrapper.o_true_confidence = true_conf

        if verbose:
            print(f"\ttrue conf: {true_conf}")
            print("\t")
        true_pca_conf_s_to_o: float = get_pca_confidence_on_observed_data(
            predictions_as_pylo_const_tuple_set=predictions_as_pylo_const_tuple_set,
            true_entity_tuple_set=true_entity_tuple_set,
            pca_non_target_entity_set=pca_subject_set,
            predict_object_entity=True,
            verbose=verbose
        )
        rule_wrapper.o_true_pca_confidence_subject_to_object = true_pca_conf_s_to_o

        print("\n----\n")

        true_pca_conf_o_to_s: float = get_pca_confidence_on_observed_data(
            predictions_as_pylo_const_tuple_set=predictions_as_pylo_const_tuple_set,
            true_entity_tuple_set=true_entity_tuple_set,
            pca_non_target_entity_set=pca_object_set,
            predict_object_entity=False,
            verbose=verbose
        )
        rule_wrapper.o_true_pca_confidence_object_to_subject = true_pca_conf_o_to_s
    else:
        pass  # no predictions

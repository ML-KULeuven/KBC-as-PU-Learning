import gzip
import json
from enum import Enum
from typing import Optional, Dict, Union, Iterator, Set, List

from problog.logic import Clause as ProblogClause, Term as ProblogTerm

from pylo.language.lp import Clause as PyloClause, Literal as PyloLiteral
from pylo.language.lp import global_context as pylo_global_context

import pandas as pd

from kbc_pul.amie.amie_rule_string_to_problog_clause_conversion import DatalogAMIERuleConvertor
from kbc_pul.prolog_utils.problog_to_pylo_conversion import convert_clause

from kbc_pul.confidence_naming import ConfidenceEnum


class AmieOutputKeyEnum(Enum):
    RULE = "Rule"
    HEAD_COVERAGE = 'Head Coverage'  # rule support / head relation size in observed KB
    STD_CONF = 'Std Confidence'
    PCA_CONF = 'PCA Confidence'
    N_POSITIVE_EXAMPLES = 'Positive Examples'  # rule support, i.e. nb of predictions supported by the rule
    BODY_SIZE = 'Body size'
    PCA_BODY_SIZE = 'PCA Body size'
    FUNCTIONAL_VARIABLE = 'Functional variable'


columns_header_without_amie = ["Rule", 'Nb supported predictions', 'Body size'] + [
    conf.get_name()
    for conf in ConfidenceEnum
]


class RuleWrapper:
    def __init__(self, rule: PyloClause, o_amie_dict: Optional[Dict] = None):
        """

        :param rule:
        :param o_amie_dict:
        """
        self.rule: PyloClause = rule

        self.amie_dict: Optional[Dict] = o_amie_dict

        # BODY support
        # # predictions
        self.o_n_predictions: Optional[int] = None  # a.k.a. the body support
        # RULE support
        # # predictions in the observed KB
        self.o_n_supported_predictions: Optional[int] = None  # a.k.a the 'Positive predictions' from AMIE

        self.o_std_confidence: Optional[float] = None
        self.o_pca_confidence_subject_to_object: Optional[float] = None
        self.o_pca_confidence_object_to_subject: Optional[float] = None

        self.o_c_weighted_std_conf: Optional[float] = None

        self.o_relative_pu_confidence_unbiased: Optional[float] = None

        self.o_relative_pu_confidence_pca_subject_to_object: Optional[float] = None
        self.o_relative_pu_confidence_pca_object_to_subject: Optional[float] = None

        # self.o_absolute_pu_confidence: Optional[float] = None

        self.o_true_confidence: Optional[float] = None
        self.o_true_pca_confidence_subject_to_object: Optional[float] = None
        self.o_true_pca_confidence_object_to_subject: Optional[float] = None

    def get_value(self, conf_name: ConfidenceEnum):

        if conf_name is ConfidenceEnum.CWA_CONF:
            return self.o_std_confidence
        elif conf_name is ConfidenceEnum.ICW_CONF:
            return self.o_c_weighted_std_conf

        elif conf_name is ConfidenceEnum.PCA_CONF_S_TO_O:
            return self.o_pca_confidence_subject_to_object
        elif conf_name is ConfidenceEnum.PCA_CONF_O_TO_S:
            return self.o_pca_confidence_object_to_subject

        elif conf_name is ConfidenceEnum.IPW_CONF:
            return self.o_relative_pu_confidence_unbiased
        elif conf_name is ConfidenceEnum.IPW_PCA_CONF_S_TO_O:
            return self.o_relative_pu_confidence_pca_subject_to_object
        elif conf_name is ConfidenceEnum.IPW_PCA_CONF_O_TO_S:
            return self.o_relative_pu_confidence_pca_object_to_subject

        elif conf_name is ConfidenceEnum.TRUE_CONF:
            return self.o_true_confidence
        elif conf_name is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_S_TO_O:
            return self.o_true_pca_confidence_subject_to_object
        elif conf_name is ConfidenceEnum.TRUE_CONF_BIAS_YS_ZERO_O_TO_S:
            return self.o_true_pca_confidence_object_to_subject

        else:
            raise Exception(f"Could not find RuleWrapper instance variable corresponding to {conf_name}")

    @staticmethod
    def create_rule_wrapper(
            rule: PyloClause,
            amie_dict: Optional[Dict],
            o_n_predictions: Optional[int],
            o_n_supported_predictions: Optional[int],
            o_std_confidence: Optional[float],

            o_pca_confidence_subject_to_object: Optional[float],
            o_pca_confidence_object_to_subject: Optional[float],

            o_c_weighted_std_conf: Optional[float],

            o_relative_pu_confidence_unbiased: Optional[float],

            o_relative_pu_confidence_pca_subject_to_object: Optional[float],
            o_relative_pu_confidence_pca_object_to_subject: Optional[float],

            o_true_confidence: Optional[float],
            o_true_pca_confidence_subject_to_object: Optional[float],
            o_true_pca_confidence_object_to_subject: Optional[float],

    ) -> 'RuleWrapper':
        new_rule_wrapper: RuleWrapper = RuleWrapper(rule=rule, o_amie_dict=amie_dict)
        new_rule_wrapper.o_n_predictions = o_n_predictions
        new_rule_wrapper.o_n_supported_predictions = o_n_supported_predictions
        new_rule_wrapper.o_std_confidence = o_std_confidence
        new_rule_wrapper.o_pca_confidence_subject_to_object = o_pca_confidence_subject_to_object
        new_rule_wrapper.o_pca_confidence_object_to_subject = o_pca_confidence_object_to_subject

        new_rule_wrapper.o_c_weighted_std_conf = o_c_weighted_std_conf

        new_rule_wrapper.o_relative_pu_confidence_unbiased = o_relative_pu_confidence_unbiased
        new_rule_wrapper.o_relative_pu_confidence_pca_subject_to_object = o_relative_pu_confidence_pca_subject_to_object
        new_rule_wrapper.o_relative_pu_confidence_pca_object_to_subject = o_relative_pu_confidence_pca_object_to_subject

        new_rule_wrapper.o_true_confidence = o_true_confidence
        new_rule_wrapper.o_true_pca_confidence_subject_to_object = o_true_pca_confidence_subject_to_object
        new_rule_wrapper.o_true_pca_confidence_object_to_subject = o_true_pca_confidence_object_to_subject

        return new_rule_wrapper

    def set_inverse_c_weighted_std_confidence(self, label_frequency: float) -> None:
        if label_frequency == 0.0:
            raise Exception("Label frequency cannot be zero")
        self.o_c_weighted_std_conf = 1 / label_frequency * self.o_std_confidence

    def instantiate_from_amie_dict(self, o_amie_dict: Optional[Dict] = None) -> None:

        if o_amie_dict is None:
            amie_dict_to_use: Optional[Dict] = self.amie_dict
        else:
            amie_dict_to_use = o_amie_dict

        if amie_dict_to_use is None:
            raise Exception(f"No AMIE dict available for rule wrapper {str(self)}")
        else:
            if self.o_n_predictions is not None:
                print("overwriting n_predictions")
            self.o_n_predictions = amie_dict_to_use[AmieOutputKeyEnum.BODY_SIZE.value]

            if self.o_n_supported_predictions is not None:
                print("overwriting o_n_supported_predictions")
            self.o_n_supported_predictions = amie_dict_to_use[AmieOutputKeyEnum.N_POSITIVE_EXAMPLES.value]

            if self.o_std_confidence is not None:
                print("overwriting o_std_confidence")
            self.o_std_confidence = amie_dict_to_use[AmieOutputKeyEnum.STD_CONF.value]

            if amie_dict_to_use[AmieOutputKeyEnum.FUNCTIONAL_VARIABLE.value] == '?a':
                if self.o_pca_confidence_subject_to_object is not None:
                    print("overwriting o_pca_confidence_subject_to_object")
                self.o_pca_confidence_subject_to_object = amie_dict_to_use[AmieOutputKeyEnum.PCA_CONF.value]
                print("ONLY value for o_pca_confidence_subject_to_object")
                print("NO value for o_pca_confidence_object_to_subject")
            elif amie_dict_to_use[AmieOutputKeyEnum.FUNCTIONAL_VARIABLE.value] == '?b':
                if self.o_pca_confidence_object_to_subject is not None:
                    print("overwriting o_pca_confidence_object_to_subject")
                self.o_pca_confidence_object_to_subject = amie_dict_to_use[AmieOutputKeyEnum.PCA_CONF.value]
                print("ONLY value for o_pca_confidence_object_to_subject")
                print("NO value for o_pca_confidence_subject_to_object")
            else:
                raise Exception(f"Unrecognized functional AMIE variabele: "
                                f"{amie_dict_to_use[AmieOutputKeyEnum.FUNCTIONAL_VARIABLE.value]}")

    def to_json_file(self, filename: str, gzipped: bool = True):

        dict_to_convert: Dict = dict()
        rule_str = str(self.rule)
        dict_to_convert['rule'] = rule_str

        if self.amie_dict is not None:
            dict_to_convert['amie_dict'] = self.amie_dict

        if self.o_n_predictions is not None:
            dict_to_convert['o_n_predictions'] = self.o_n_predictions

        if self.o_n_supported_predictions is not None:
            dict_to_convert['o_n_supported_predictions'] = self.o_n_supported_predictions

        if self.o_std_confidence is not None:
            dict_to_convert['o_std_confidence'] = self.o_std_confidence

        if self.o_pca_confidence_subject_to_object is not None:
            dict_to_convert['o_pca_confidence_subject_to_object'] = self.o_pca_confidence_subject_to_object

        if self.o_pca_confidence_object_to_subject is not None:
            dict_to_convert['o_pca_confidence_object_to_subject'] = self.o_pca_confidence_object_to_subject

        if self.o_c_weighted_std_conf is not None:
            dict_to_convert['o_c_weighted_std_conf'] = self.o_c_weighted_std_conf

        if self.o_relative_pu_confidence_unbiased is not None:
            dict_to_convert['o_relative_pu_confidence_unbiased'] = self.o_relative_pu_confidence_unbiased

        if self.o_relative_pu_confidence_pca_subject_to_object is not None:
            dict_to_convert[
                'o_relative_pu_confidence_pca_subject_to_object'
            ] = self.o_relative_pu_confidence_pca_subject_to_object

        if self.o_relative_pu_confidence_pca_object_to_subject is not None:
            dict_to_convert[
                'o_relative_pu_confidence_pca_object_to_subject'
            ] = self.o_relative_pu_confidence_pca_object_to_subject

        # if self.o_absolute_pu_confidence is not None:
        #     dict_to_convert['o_absolute_pu_confidence'] = self.o_absolute_pu_confidence

        if self.o_true_confidence is not None:
            dict_to_convert['o_true_confidence'] = self.o_true_confidence

        if self.o_true_pca_confidence_subject_to_object is not None:
            dict_to_convert['o_true_pca_confidence_subject_to_object'] = self.o_true_pca_confidence_subject_to_object

        if self.o_true_pca_confidence_object_to_subject is not None:
            dict_to_convert['o_true_pca_confidence_object_to_subject'] = self.o_true_pca_confidence_object_to_subject

        pretty_frozen = json.dumps(dict_to_convert, indent=2, sort_keys=True)
        if gzipped:
            open_func = gzip.open
        else:
            open_func = open

        with open_func(filename, 'wt') as output_file:
            output_file.write(pretty_frozen)

    @staticmethod
    def read_json(filename: str, gzipped: bool = True) -> 'RuleWrapper':

        if gzipped:
            open_func = gzip.open
        else:
            open_func = open

        with open_func(filename, 'rt') as input_file:
            dict_to_convert: Dict = json.load(input_file)
            rule_str: str = dict_to_convert["rule"]

            problog_rule: ProblogClause = ProblogTerm.from_string(rule_str)
            pylo_rule: Union[PyloLiteral, PyloClause] = convert_clause(
                clause=problog_rule,
                context=pylo_global_context
            )

            rule_wrapper = RuleWrapper.create_rule_wrapper(
                rule=pylo_rule,
                amie_dict=dict_to_convert.get('amie_dict', None),

                o_n_predictions=dict_to_convert.get('o_n_predictions', None),
                o_n_supported_predictions=dict_to_convert.get('o_n_supported_predictions', None),

                o_std_confidence=dict_to_convert.get('o_std_confidence', None),

                o_pca_confidence_subject_to_object=dict_to_convert.get(
                    'o_pca_confidence_subject_to_object', None),
                o_pca_confidence_object_to_subject=dict_to_convert.get(
                    'o_pca_confidence_object_to_subject', None),

                o_c_weighted_std_conf=dict_to_convert.get('o_c_weighted_std_conf', None),

                o_relative_pu_confidence_unbiased=dict_to_convert.get(
                    'o_relative_pu_confidence_unbiased', None),

                o_relative_pu_confidence_pca_subject_to_object=dict_to_convert.get(
                    'o_relative_pu_confidence_pca_subject_to_object', None),
                o_relative_pu_confidence_pca_object_to_subject=dict_to_convert.get(
                    'o_relative_pu_confidence_pca_object_to_subject', None),

                # o_absolute_pu_confidence = dict_to_convert.get('o_absolute_pu_confidence', None)

                o_true_confidence=dict_to_convert.get('o_true_confidence', None),
                o_true_pca_confidence_subject_to_object=dict_to_convert.get(
                    'o_true_pca_confidence_subject_to_object', None),
                o_true_pca_confidence_object_to_subject=dict_to_convert.get(
                    'o_true_pca_confidence_object_to_subject', None),
            )

            return rule_wrapper

    def clone_with_metrics_unset(self) -> 'RuleWrapper':
        return RuleWrapper(rule=self.rule)

    def clone(self) -> 'RuleWrapper':
        return RuleWrapper.create_rule_wrapper(
            rule=self.rule,
            amie_dict=self.amie_dict,

            o_n_predictions=self.o_n_predictions,
            o_n_supported_predictions=self.o_n_supported_predictions,

            o_std_confidence=self.o_std_confidence,

            o_pca_confidence_subject_to_object=self.o_pca_confidence_subject_to_object,
            o_pca_confidence_object_to_subject=self.o_pca_confidence_object_to_subject,

            o_c_weighted_std_conf=self.o_c_weighted_std_conf,

            o_relative_pu_confidence_unbiased=self.o_relative_pu_confidence_unbiased,

            o_relative_pu_confidence_pca_subject_to_object=self.o_relative_pu_confidence_pca_subject_to_object,
            o_relative_pu_confidence_pca_object_to_subject=self.o_relative_pu_confidence_pca_object_to_subject,

            o_true_confidence=self.o_true_confidence,
            o_true_pca_confidence_subject_to_object=self.o_true_pca_confidence_subject_to_object,
            o_true_pca_confidence_object_to_subject=self.o_true_pca_confidence_object_to_subject,

        )

    @property
    def o_rule_support(self) -> Optional[int]:
        """
        The support of the rule =
            the number of predictions in the observed KB.

        :return:
        """
        return self.o_n_supported_predictions

    @property
    def o_body_support(self) -> Optional[int]:
        """
        The support of the rule BODY =
            the number of predictions of the rule.

        :return:
        """
        return self.o_n_predictions

    def __str__(self):
        if self.amie_dict is not None:
            return str(self.amie_dict)
        else:
            return str(self.rule)

    def __repr__(self):
        return self.__str__()

    def get_columns_header(self) -> List[str]:
        header = self.get_columns_header_without_amie()
        if self.amie_dict is not None:
            header = header + [key.value + " (AMIE)" for key in AmieOutputKeyEnum]
        return header

    @staticmethod
    def get_columns_header_without_amie() -> List[str]:
        return columns_header_without_amie

    def to_row(self, include_amie_metrics: bool = True) -> List[Union[str, float]]:
        row = [
                  str(self.rule),
                  self.o_n_supported_predictions,
                  self.o_n_predictions
              ] + [self.get_value(conf) for conf in ConfidenceEnum]

        if include_amie_metrics and self.amie_dict is not None:
            row.extend([self.amie_dict[key.value] for key in AmieOutputKeyEnum])
        return row

    def get_amie_rule_string_repr(self) -> str:
        """
        :return: the string representation of the rule as generated by AMIE
        """
        if self.amie_dict is None:
            raise Exception()
        else:
            return self.amie_dict[AmieOutputKeyEnum.RULE.value]

    def get_amie_head_coverage(self) -> float:
        """
                                    support of the rule = # observed predictions
        Head coverage of a rule =  ------------------------------------------------------------
                                    number of literals with the head relation in the observed KB

        hc(r) =  \frac{ supp(r) }{ #(x,y): h(x,y) }


        The support of a rule (= the rule's observed predictions) is an integer number.
        Head coverage gives a relative version by dividing it by
         the number of literals with the head's relation in the observed KB.

        :return: the head coverage according AMIE
        """
        if self.amie_dict is None:
            raise Exception()
        else:
            return self.amie_dict[AmieOutputKeyEnum.HEAD_COVERAGE.value]

    def get_amie_rule_support(self) -> int:
        """
        Rule support = nb of predictions done by the rule

        :return:
        """
        pass

    @staticmethod
    def create_rule_wrapper_from(amie_output_rule_series: pd.Series) -> 'RuleWrapper':
        amie_rule_dict: Dict = amie_output_rule_series.to_dict()
        rule_string: str = amie_rule_dict[AmieOutputKeyEnum.RULE.value]

        problog_rule: ProblogClause = DatalogAMIERuleConvertor.convert_amie_datalog_rule_string_to_problog_clause(
            rule_string)

        pylo_rule: Union[PyloLiteral, PyloClause] = convert_clause(
            clause=problog_rule,
            context=pylo_global_context
        )
        return RuleWrapper(rule=pylo_rule, o_amie_dict=amie_rule_dict)

    def set_std_confidence(self, calculated_value: float) -> None:
        if self.amie_dict is not None:
            amie_std_conf_value: float = self.amie_dict[AmieOutputKeyEnum.STD_CONF.value]
            if abs(amie_std_conf_value - calculated_value) >= 0.01:
                print(f"Calculated STD conf differs from AMIE: {calculated_value:0.3f} vs {amie_std_conf_value:0.3f}")
        self.o_std_confidence = calculated_value

    def set_pca_confidence(self, calculated_value: float) -> None:
        if self.amie_dict is not None:
            amie_pca_conf_value: float = self.amie_dict[AmieOutputKeyEnum.PCA_CONF.value]
            if abs(amie_pca_conf_value - calculated_value) >= 0.01:
                print(f"Calculated PCA conf differs from AMIE: {calculated_value:0.3f} vs {amie_pca_conf_value:0.3f}")
        self.o_pca_confidence_subject_to_object = calculated_value


def get_pylo_rule_from_string(rule_str: str) -> PyloClause:
    problog_rule: ProblogClause = ProblogTerm.from_string(rule_str)
    pylo_rule: Union[PyloLiteral, PyloClause] = convert_clause(
        clause=problog_rule,
        context=pylo_global_context
    )
    return pylo_rule


def is_pylo_rule_recursive(pylo_rule: PyloClause) -> bool:
    head_functor: str = pylo_rule.get_head().get_predicate().get_name()

    body_functors: Set[str] = {
        body_literal.get_predicate().get_name()
        for body_literal in pylo_rule.get_body().get_literals()
    }
    return head_functor in body_functors


def filter_rules_predicting(
        rule_wrapper_sequence: Iterator[RuleWrapper],
        head_functor_set: Optional[Set[str]] = None
) -> List[RuleWrapper]:
    # filtered_rules: List[RuleWrapper] = []
    # rule_wrapper: RuleWrapper
    # for rule_wrapper in rule_wrapper_sequence:
    #     if rule_wrapper.rule.get_head().predicate.name in head_functor_set:
    #         filtered_rules.append(rule_wrapper)

    if head_functor_set is None:
        return [rule_wrapper for rule_wrapper in rule_wrapper_sequence]
    else:
        return [rule_wrapper for rule_wrapper in rule_wrapper_sequence
                if rule_wrapper.rule.get_head().predicate.name in head_functor_set]


def contains_rule_predicting_relation(
        rule_wrapper_sequence: Iterator[RuleWrapper],
        target_relation: str
) -> bool:
    return len(
        filter_rules_predicting(
            rule_wrapper_sequence=rule_wrapper_sequence,
            head_functor_set={target_relation}
        )
    ) > 0


def create_amie_dataframe_from_rule_wrappers(rule_wrapper_collection: List[RuleWrapper]) -> pd.DataFrame:
    columns = [AmieOutputKeyEnum.RULE.value] + [key for key in rule_wrapper_collection[0].amie_dict.keys()
                                                if key != AmieOutputKeyEnum.RULE.value]
    data: List[List] = []
    for rule in rule_wrapper_collection:
        row = [str(rule.rule)] + [rule.amie_dict[key] for key in columns if key != AmieOutputKeyEnum.RULE.value]
        data.append(row)
    df = pd.DataFrame(data=data, columns=columns)
    return df


def create_extended_dataframe_from_rule_wrappers(rule_wrapper_collection: List[RuleWrapper]) -> pd.DataFrame:
    columns_header = rule_wrapper_collection[0].get_columns_header()
    row_data = []
    rule_wrapper: RuleWrapper
    for rule_wrapper in rule_wrapper_collection:
        row = rule_wrapper.to_row()
        row_data.append(row)
    df = pd.DataFrame(data=row_data, columns=columns_header)
    return df


def create_dataframe_without_amie_from_rule_wrappers(rule_wrapper_collection: List[RuleWrapper]) -> pd.DataFrame:
    columns_header = RuleWrapper.get_columns_header_without_amie()
    row_data = []
    rule_wrapper: RuleWrapper
    for rule_wrapper in rule_wrapper_collection:
        row = rule_wrapper.to_row(include_amie_metrics=False)
        row_data.append(row)
    df = pd.DataFrame(data=row_data, columns=columns_header)
    return df

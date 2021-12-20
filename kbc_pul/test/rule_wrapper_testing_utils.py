from typing import Union, Dict

from kbc_pul.data_structures.rule_wrapper import RuleWrapper
from problog_to_pylo.conversion import convert_clause

from pylo.language.lp import Clause as PyloClause, Literal as PyloLiteral
from pylo.language.lp import global_context as pylo_global_context

from problog.logic import Clause as ProblogClause, Term as ProblogTerm


def get_rule_wrapper_from_str_repr(rule_wrapper_prolog_string: str) -> RuleWrapper:
    problog_rule: ProblogClause = ProblogTerm.from_string(rule_wrapper_prolog_string)

    pylo_rule: Union[PyloLiteral, PyloClause] = convert_clause(
        clause=problog_rule,
        context=pylo_global_context
    )
    amie_dict: Dict = dict()
    return RuleWrapper(rule=pylo_rule, o_amie_dict=amie_dict)


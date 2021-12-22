# %%
from typing import List, Dict

from kbc_pul.data_structures.rule_wrapper import get_pylo_rule_from_string
from pylo.language.lp import Clause, Literal, Atom

#
# rule_name = "dealswith(A,B):-hasneighbor(G,A), hasneighbor(G,B)"


def get_paper_like_rule_string_from_prolog_str(prolog_rule_string: str) -> str:
    rule: Clause = get_pylo_rule_from_string(prolog_rule_string)
    return get_paper_like_rule_string_from_pylo_clause(rule)


def get_paper_like_rule_string_from_pylo_clause(rule: Clause) -> str:
    """

    :param rule: the rule to represent
    :return: A string representation of a rule to put in a LaTex table
    """

    body_literals: List[Literal] = rule.get_body().get_literals()
    head: Atom = rule.get_head()
    head_pred_name = head.get_predicate().get_name()

    variable_map: Dict[str, str] = dict()

    head_arguments = head.get_arguments()

    variable_map[head_arguments[0].name] = 's'
    variable_map[head_arguments[1].name] = 'o'

    new_body_lit_list = []

    for body_lit in body_literals:
        body_pred_name: str = body_lit.get_predicate().get_name()
        body_variables = body_lit.get_variables()
        old_first_var_name: str = body_variables[0].get_name()
        old_second_var_name: str = body_variables[1].get_name()

        o_new_first_var_name = variable_map.get(old_first_var_name, None)
        if o_new_first_var_name is None:
            o_new_first_var_name = old_first_var_name.lower()
            variable_map[old_first_var_name] = o_new_first_var_name

        o_new_second_var_name = variable_map.get(old_second_var_name, None)
        if o_new_second_var_name is None:
            o_new_second_var_name = old_second_var_name.lower()
            variable_map[old_second_var_name] = o_new_second_var_name

        new_lit = "\langle " + f"{o_new_first_var_name}, {body_pred_name}, {o_new_second_var_name}" + " \\rangle"
        new_body_lit_list.append(new_lit)

    new_body: str = " \wedge ".join(new_body_lit_list)
    new_head = "\langle s, " + head_pred_name + " , o \\rangle"
    new_rule_string: str = "$" + new_body + " \Rightarrow " + new_head + "$"
    return new_rule_string

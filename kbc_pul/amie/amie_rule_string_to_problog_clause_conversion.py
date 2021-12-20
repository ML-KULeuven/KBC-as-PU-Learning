"""


"""


from typing import Union, Optional, List, Tuple
import re
from problog.logic import Clause, Constant, Var, Term, And

from kbc_pul.prolog_utils.problog_logic_manipulation_utils import TermManipulationUtils


class AMIERuleConvertor:
    @staticmethod
    def convert_amie_rule_string_to_problog_clause(amie_rule_string: str) -> Clause:
        raise NotImplementedError('abstract method')

    @classmethod
    def split_amie_datalog_rule_string_in_body_and_head_strings(cls, rule_string: str) -> Tuple[str, str]:
        body_head_list: List[str] = rule_string.split("=>")
        if len(body_head_list) != 2:
            raise Exception(f"body_head_list should have length 2, but is: {str(body_head_list)}")

        # strip leading & trailing whitespace
        body_string = body_head_list[0].strip()
        head_string = body_head_list[1].strip()
        return body_string, head_string

    @classmethod
    def convert_amie_literal_argument(cls, amie_arg: str) -> Union[Constant, Var]:
        if amie_arg[0] == '?':
            # the arg is a variable
            return Var(amie_arg[1].upper() + amie_arg[2:])
        else:
            return Constant(amie_arg)

    @classmethod
    def convert_amie_literal_to_problog_term(cls, amie_functor: str, amie_arg1: str, amie_arg2) -> Term:
        if amie_functor[0] == "<":
            raise Exception(
                f"Cannot currently deal with escaped AMIE tokens (i.e. surrounded by <...> : {amie_functor}")

        # Parse the arguments
        lit_arg_1_term: Term = cls.convert_amie_literal_argument(amie_arg1)
        lit_arg_2_term: Term = cls.convert_amie_literal_argument(amie_arg2)
        return Term(amie_functor)(lit_arg_1_term, lit_arg_2_term)


class DatalogAMIERuleConvertor:
    binary_literal_re_pattern = re.compile("(.*)\((.*),(.*)\)")

    @classmethod
    def convert_amie_datalog_literal_string_to_term(cls, literal_string: str) -> Term:
        o_match: Optional[re.Match] = cls.binary_literal_re_pattern.match(literal_string)
        if o_match is not None:
            lit_functor_str, lit_arg1_str, lit_arg2_str = o_match.group(1, 2, 3)
            literal_term: Term = AMIERuleConvertor.convert_amie_literal_to_problog_term(
                lit_functor_str, lit_arg1_str, lit_arg2_str)
        else:
            raise Exception(f"could not parse {literal_string}")
        return literal_term

    @classmethod
    def convert_amie_datalog_rule_string_to_problog_clause(cls, rule_string: str) -> Clause:
        body_string: str
        head_string: str
        body_string, head_string = AMIERuleConvertor.split_amie_datalog_rule_string_in_body_and_head_strings(
            rule_string)

        # parse body
        # 1. split in literal strings
        body_literal_strings_list: List[str] = body_string.split(" ")

        # 2. for each body literal, parse the functor, arg1 & arg2
        body_terms_list: List[Term] = []
        literal_string: str
        for literal_string in body_literal_strings_list:
            literal_term = cls.convert_amie_datalog_literal_string_to_term(literal_string)
            body_terms_list.append(literal_term)
        # 3. convert list of Terms to Problog conjunction
        body_as_conj: And = TermManipulationUtils.list_to_conjunction(body_terms_list)

        # 4. Convert the head string to a Problog Term
        head_term: Term = cls.convert_amie_datalog_literal_string_to_term(head_string)

        # 5. Create a Problog rule
        problog_rule: Clause = Clause(head_term, body_as_conj)
        return problog_rule


class SparqlAMIERuleConvertor:

    @classmethod
    def convert_amie_sparql_rule_string_to_problog_clause(cls, rule_string: str) -> Clause:
        body_string: str
        head_string: str
        body_string, head_string = AMIERuleConvertor.split_amie_datalog_rule_string_in_body_and_head_strings(
            rule_string)

        # parse body
        # 1. split in body tokens
        body_token_strings_list: List[str] = [token for token in body_string.split(" ") if len(token) != 0]

        # 2. for each body literal, parse the functor, arg1 & arg2
        body_terms_list: List[Term] = []
        for body_token_index in range(0, len(body_token_strings_list), 3):
            body_lit_arg1_str: str = body_token_strings_list[body_token_index]
            body_lit_functor_str: str = body_token_strings_list[body_token_index + 1]
            body_lit_arg2_str: str = body_token_strings_list[body_token_index + 2]

            literal_term = AMIERuleConvertor.convert_amie_literal_to_problog_term(
                body_lit_functor_str, body_lit_arg1_str, body_lit_arg2_str)
            body_terms_list.append(literal_term)

        # 3. convert list of Terms to Problog conjunction
        body_as_conj: And = TermManipulationUtils.list_to_conjunction(body_terms_list)

        # 4. Convert the head string to a Problog Term

        head_token_string_list: List[str] = [token for token in head_string.split(" ") if len(token) != 0]
        head_arg1 = head_token_string_list[0]
        head_functor = head_token_string_list[1]
        head_arg2 = head_token_string_list[2]
        head_term: Term = AMIERuleConvertor.convert_amie_literal_to_problog_term(head_functor, head_arg1, head_arg2)

        # 5. Create a Problog rule
        problog_rule: Clause = Clause(head_term, body_as_conj)
        return problog_rule


if __name__ == '__main__':
    example_rule_string_datalog_vars_only: str = 'isa(?g,?b) uses(?g,?a)  => isa(?a,?b)'
    example_rule_string_datalog_also_consts: str = 'isa(?a,animal)  => issue_in(?a,biomedical_occupation_or_discipline)'
    example_rule_string_default_vars_only: str = '?g  isa  ?a  ?g  manifestation_of  ?b   => ?a  affects  ?b'
    example_rule_string_default_also_consts: str = '?a  interacts_with  steroid   => ?a  issue_in  occupation_or_disciplin'

    print("datalog test:")
    print("\tamie rule without constants:")
    print("\t\t", example_rule_string_datalog_vars_only)
    print("\tproblog version")
    problog_rule = DatalogAMIERuleConvertor.convert_amie_datalog_rule_string_to_problog_clause(
        example_rule_string_datalog_vars_only)
    print("\t\t", str(problog_rule))
    print()

    print("\tamie rule WITH constants:")
    print("\t\t", example_rule_string_datalog_also_consts)
    print("\tproblog version")
    problog_rule = DatalogAMIERuleConvertor.convert_amie_datalog_rule_string_to_problog_clause(
        example_rule_string_datalog_also_consts)
    print("\t\t", str(problog_rule))

    print("sparql test:")
    print("\tamie rule without constants:")
    print("\t\t", example_rule_string_default_vars_only)
    print("\tproblog version")
    problog_rule = SparqlAMIERuleConvertor.convert_amie_sparql_rule_string_to_problog_clause(
        example_rule_string_default_vars_only)
    print("\t\t", str(problog_rule))
    print()

    print("\tamie rule WITH constants:")
    print("\t\t", example_rule_string_default_also_consts)
    print("\tproblog version")
    problog_rule = SparqlAMIERuleConvertor.convert_amie_sparql_rule_string_to_problog_clause(
        example_rule_string_default_also_consts)
    print("\t\t", str(problog_rule))

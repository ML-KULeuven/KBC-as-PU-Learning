from typing import List, Union, Tuple, Dict

from problog.program import PrologString, LogicProgram

from problog.logic import Constant as ProblogConstant
from problog.logic import Var as ProblogVariable
from problog.logic import Term as ProblogTerm
from problog.logic import And as ProblogAnd
from problog.logic import Not as ProblogNot
from problog.logic import Clause as ProblogClause

from pylo.language.lp import Constant as PyloConstant
from pylo.language.lp import Variable as PyloVariable
from pylo.language.lp import Term as PyloTerm
from pylo.language.lp import Functor as PyloFunctor
from pylo.language.lp import Structure as PyloStructuredTerm
from pylo.language.lp import List as PyloList
from pylo.language.lp import Body as PyloBody
from pylo.language.lp import Clause as PyloClause
from pylo.language.lp import Predicate as PyloPredicate
from pylo.language.lp import Literal as PyloLiteral
from pylo.language.lp import Not as PyloNot
from pylo.language.lp import Context as PyloContext
from pylo.language.lp import Type as PyloType


from kbc_pul.prolog_utils.pylo_utils import is_valid_constant, is_valid_variable


class ConversionError(Exception):
    @staticmethod
    def check_type(thing_to_convert, expected_type):
        if not isinstance(thing_to_convert, expected_type):
            raise ConversionError(
                f"{thing_to_convert} has type {str(type(thing_to_convert))}, but {expected_type} was expected"
            )


def convert_constant(c: Union[ProblogConstant, ProblogTerm], context: PyloContext) -> PyloConstant:
    if isinstance(c, ProblogConstant):
        problog_value: Union[str, float, int] = c.value
        pylo_value = str(problog_value)
        return context.constant(name=pylo_value)
    else:
        functor: str = c.functor
        arity: int = int(c.arity)
        if arity == 0 and (is_valid_constant(functor)):
            return context.constant(name=functor)
        else:
            raise Exception("unchecked case")


def convert_variable(v: Union[ProblogVariable, ProblogTerm]) -> PyloVariable:
    if isinstance(v, ProblogVariable):
        functor: str = v.functor
        return PyloVariable(name=functor)
    else:
        functor: str = v.functor
        arity: int = int(v.arity)
        if arity == 0 and (functor[0].isupper() or functor[0] == "_"):
            return PyloVariable(name=functor)


def convert_problog_list(term: ProblogTerm) -> List[PyloTerm]:

    elements: List[PyloTerm] = list()

    current: ProblogTerm = term
    while current.functor != '[]':
        first_arg = current.args[0]
        converted_term: PyloTerm = convert_term(first_arg)
        elements.append(converted_term)
        current = current.args[1]
    return elements


def convert_structured_term(term: ProblogTerm, context: PyloContext) -> PyloStructuredTerm:
    ConversionError.check_type(term, ProblogTerm)
    functor: str = term.functor
    arity: str = term.arity
    arity_as_int = int(arity)
    args_tuple: Tuple[ProblogConstant, ProblogVariable, ProblogTerm] = term.args

    if functor == '.':
        # treat this as a list
        elements: List[PyloTerm] = convert_problog_list(term)
        pylo_list: PyloList = PyloList(elements=elements)
        return pylo_list
    else:
        converted_args = tuple([convert_term(arg, context) for arg in args_tuple])

        return PyloStructuredTerm(
            functor=PyloFunctor(name=functor, arity=arity_as_int),
            arguments=converted_args
        )


def convert_term(term: Union[ProblogTerm, int], context: PyloContext) -> Union[PyloTerm, int]:
    """
    A first order term, for example 'p(X,Y)'.
    """
    if isinstance(term, int):
        return term
    else:
        ConversionError.check_type(term, ProblogTerm)
        functor: str = term.functor
        arity: int = int(term.arity)

        if isinstance(functor, int) or isinstance(functor, float):
            return convert_constant(term, context)
        if arity == 0:
            if is_valid_constant(functor):
                return convert_constant(term, context)
            elif is_valid_variable(functor):
                return convert_variable(term)
            else:
                raise Exception(f"Error in converting term with arity 0: {str(term)}")
        elif isinstance(term, ProblogTerm):
            return convert_structured_term(term, context)
        else:
            raise Exception("unchecked case!")


def convert_fact(fact: ProblogTerm, context: PyloContext) -> PyloLiteral:
    return _convert_positive_literal(fact, context)


def _convert_positive_literal(literal: ProblogTerm, context: PyloContext) -> PyloLiteral:
    ConversionError.check_type(literal, ProblogTerm)
    functor: str = literal.functor
    arity: str = literal.arity
    arity_as_int = int(arity)
    args_tuple: Tuple[Union[ProblogConstant, ProblogVariable, ProblogTerm], ...] = literal.args

    converted_args = [convert_term(arg, context) for arg in args_tuple]
    predicate: PyloPredicate = context.predicate(name=functor, arity=arity_as_int)
    literal = predicate(*converted_args)
    return literal


# def _convert_negative_literal(literal: ProblogTerm, context: PyloContext):
#     functor: str = literal.functor
#     if functor != '\\+':
#         raise Exception(f"Incorrect functor {functor}, \\+ was expectd for a Not problog term")
#     args: Tuple[ProblogTerm] = literal.args
#     if len(args) != 1:
#         raise Exception(f"unexpected nb of arguments: {args}")
#     inner_problog_term: ProblogTerm = args[0]
#     converted_literl


def convert_body_literal(literal: ProblogTerm, context: PyloContext) -> Union[PyloLiteral, PyloNot]:
    functor = literal.functor
    if isinstance(literal, ProblogNot) or functor == '\\+':
        args: Tuple[ProblogTerm] = literal.args
        if len(args) != 1:
            raise Exception(f"unexpected nb of arguments: {args}")
        inner_problog_term: ProblogTerm = args[0]
        converted_inner_literal: Union[PyloLiteral, PyloNot] = convert_body_literal(inner_problog_term, context)
        return PyloNot(converted_inner_literal)
    else:
        return _convert_positive_literal(literal, context)


def _convert_problog_body(body: ProblogTerm, context: PyloContext) -> PyloBody:
    if isinstance(body, ProblogAnd):
        body_terms: List[ProblogTerm] = body.to_list()
        pylo_body_literals: List[Union[PyloLiteral, PyloNot]] = [
            convert_body_literal(term, context) for term in body_terms
        ]
        pylo_body: PyloBody = PyloBody(*pylo_body_literals)
    else:
        converted_body_literal: Union[PyloLiteral, PyloNot] = convert_body_literal(body, context)
        pylo_body: PyloBody = PyloBody(converted_body_literal)
    return pylo_body


def convert_clause_with_head_and_body(clause: ProblogClause, context: PyloContext) -> PyloClause:
    functor = clause.functor
    if functor != ":-":
        raise Exception("unexpected input")
    head: ProblogTerm = clause.head
    body: ProblogTerm = clause.body

    pylo_head: PyloLiteral = _convert_positive_literal(head, context)
    pylo_body: PyloBody = _convert_problog_body(body=body, context=context)

    pylo_clause = PyloClause(head=pylo_head, body=pylo_body)
    return pylo_clause


def convert_clause(clause: Union[ProblogClause, ProblogTerm], context: PyloContext) -> Union[PyloLiteral, PyloClause]:
    if isinstance(clause, ProblogClause):
        return convert_clause_with_head_and_body(clause, context)
    elif isinstance(clause, ProblogTerm):
        return convert_fact(clause, context)
    else:
        raise Exception(f"unchecked case! \n\t{str(clause)}\n\t{type(clause)}")


def convert_logic_program(problog_program: LogicProgram, context: PyloContext) -> List[Union[PyloLiteral, PyloClause]]:
    pylo_clauses: List[Union[PyloLiteral, PyloClause]] = []
    for clause in problog_program:
        converted_clause: Union[PyloLiteral, PyloClause] = convert_clause(clause, context=context)
        pylo_clauses.append(converted_clause)
    return pylo_clauses


def main_pylo_substitution():
    from pylo.language.lp import global_context
    problog_term: ProblogTerm = ProblogTerm.from_string("foo(X)")
    pylo_term: PyloLiteral = convert_fact(problog_term, context=global_context)

    x = global_context.variable(name="X")
    bar = global_context.constant("bar")
    substitution = {x: bar}
    new_pylo_term = pylo_term.substitute(substitution)
    print(new_pylo_term)

def main():
    logic_program = PrologString("""
testnr(1,1).
lumo(1,-1.246).
logp(1,4.23).
nitro(1,[d1_19,d1_24,d1_25,d1_26]).
dmuta(Mol, pos) :- logmutag(Mol, L), L> 0.
dmuta(Mol, neg) :- logmutag(Mol, L), L =< 0.

sbond(M,X,Y,Z) :- bond(M,X,Y,Z).
sbond(M,X,Y,Z) :- bond(M,Y,X,Z).


to_query :- bongard(A,B),triangle(A,C).
    """)
    from pylo.language.lp import global_context
    from pylo.engines.prolog.SWIProlog import SWIProlog
    pl = SWIProlog('/usr/bin/swipl')

    pylo_clauses = convert_logic_program(logic_program, global_context)
    for clause in pylo_clauses:
        pl.assertz(clause)

    # from pylo import global_context
    #

    # f = convert_fact(fact, global_context)
    # pl.assertz(f)
    #
    nitro = global_context.get_predicate("nitro", 2)
    print(pl.query(nitro("X", "Y")))


    # term1 = ProblogTerm('\'Golden_Globe_Award_for_Best_Supporting_Actress_-_Motion_Picture\'')
    # term2 = ProblogTerm('\'Olympia_Dukakis\'')


def main2():

    from pylo.language.lp import global_context
    from pylo.engines.prolog.SWIProlog import SWIProlog
    pl = SWIProlog('/usr/bin/swipl')

    problog_term = ProblogTerm('\'?\'')
    pylo_term = convert_term(problog_term)
    print(pylo_term)


def main3():
    from pylo.language.lp import global_context
    from pylo.engines.prolog.SWIProlog import SWIProlog
    pl = SWIProlog('/usr/bin/swipl')

    problog_term = ProblogTerm('set_prolog_flag')(
        ProblogConstant('stack_limit'), 1 * 1000_000_000)
    pylo_term = convert_fact(problog_term, context=global_context)
    print(pylo_term)
    pl.query(pylo_term)


if __name__ == '__main__':
    main_pylo_substitution()

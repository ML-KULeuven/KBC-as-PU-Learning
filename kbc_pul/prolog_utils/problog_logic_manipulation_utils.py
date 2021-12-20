from typing import List, Union

from problog.logic import Term, And, Not


class PartialSubstitutionDict:
    def __init__(self, partial_dict):
        self.partial_dict = partial_dict

    def __getitem__(self, key):
        self.partial_dict.get(key, key)


def _conj2list_rec(conj: Term, acc: List[Term]) -> List[Term]:
    if isinstance(conj, And):
        acc.append(conj.args[0])
        return _conj2list_rec(conj.args[1], acc)
    else:
        acc.append(conj)
        return acc


def _list2conj_rec(conj_list: List[Term], at_index: int) -> Union[And, Term]:
    if at_index == len(conj_list) - 1:
        return conj_list[-1]
    else:
        first_arg = conj_list[at_index]
        second_arg = _list2conj_rec(conj_list, at_index + 1)

        return And(first_arg, second_arg)


class TermManipulationUtils:

    @staticmethod
    def conjunction_to_list(conj: Term) -> List[Term]:
        return _conj2list_rec(conj, [])

    @staticmethod
    def list_to_conjunction(conj_list: List[Term]) -> Union[And, Term]:
        return _list2conj_rec(conj_list, 0)

    @staticmethod
    def term_is_functor_or_negation(term: Term, functor: str) -> bool:
        t = term.child if isinstance(term, Not) else term
        return t.functor == functor

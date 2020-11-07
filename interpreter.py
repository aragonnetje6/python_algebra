"""
DEPRECATED
interpreters to convert between rpn, tuple representation, and expression tree.
also has calculation tools for rpn expressions.
"""

from math import log, sin, cos, tan, asin, acos, atan
from typing import Callable
from typing import Optional, Dict, Union, List

from math_tree import Number, Variables, Node, UnaryOperator, BinaryOperator, Constant, Variable, Addition, \
    Subtraction, Product, Division, Exponent, Logarithm, Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute

operator_2_in_functions: Dict[str, Callable[[Number, Number], Number]] = {'+': lambda a, b: a + b,
                                                                          '-': lambda a, b: a - b,
                                                                          '*': lambda a, b: a * b,
                                                                          '/': lambda a, b: a / b,
                                                                          '**': lambda a, b: a ** b,
                                                                          'log': log}

operator_2_in_classes: Dict[str, Callable[[Node, Node], BinaryOperator]] = {'+': Addition,
                                                                            '-': Subtraction,
                                                                            '*': Product,
                                                                            '/': Division,
                                                                            '**': Exponent,
                                                                            'log': Logarithm}

operator_1_in_functions: Dict[str, Callable[[Number], Number]] = {'sin': sin,
                                                                  'cos': cos,
                                                                  'tan': tan,
                                                                  'asin': asin,
                                                                  'acos': acos,
                                                                  'atan': atan,
                                                                  'abs': abs}

operator_1_in_classes: Dict[str, Callable[[Node], UnaryOperator]] = {'sin': Sine,
                                                                     'cos': Cosine,
                                                                     'tan': Tangent,
                                                                     'asin': ArcSine,
                                                                     'acos': ArcCosine,
                                                                     'atan': ArcTangent,
                                                                     'abs': Absolute}

operator_functions: Dict[str, Union[Callable[[Number], Number], Callable[[Number, Number], Number]]] = \
    {**operator_1_in_functions, **operator_2_in_functions}
operator_classes: Dict[str, Union[Callable[[Node], UnaryOperator], Callable[[Node, Node], BinaryOperator]]] = \
    {**operator_1_in_classes, **operator_2_in_classes}


def rpn_to_tuple(rpn_string: str) -> tuple:
    """DEPRECATED -- convert rpn string to tuple representation of expression"""
    stack: List[Union[tuple, Number, str]] = list()
    word_list = rpn_string.split()
    for word in word_list:
        if word.isdecimal() or word[0] == '-' and word[1:].isdecimal():
            stack.append(int(word))
        elif word in operator_2_in_functions.keys():
            term2 = stack.pop()
            term1 = stack.pop()
            new_term: tuple = (term1, term2, word)
            stack.append(new_term)
        elif word in operator_1_in_functions.keys():
            term = stack.pop()
            stack.append((term, word))
        elif '.' in word or 'e' in word:
            try:
                stack.append(float(word))
            except ValueError:
                stack.append(word)
        else:
            stack.append(word)
    out = stack.pop()
    if isinstance(out, tuple) and len(stack) == 0:
        return out
    else:
        raise ValueError('invalid expression')


def tuple_to_ans(tuple_representation: tuple, var_dict: Optional[Variables] = None) -> Number:
    """DERECATED -- calculate result of tuple representation of expression"""
    if var_dict is None:
        var_dict = {}
    if isinstance(tuple_representation, (int, float)):
        return tuple_representation
    elif isinstance(tuple_representation, str):
        return var_dict[tuple_representation]
    else:
        *args, operator = tuple_representation
        return operator_functions[operator](*map(lambda x: tuple_to_ans(x, var_dict), args))


def rpn_to_ans(rpn_string: str, var_dict: Optional[Variables] = None) -> Number:
    """DEPRECATED -- calculate result of rpn expression"""
    return tuple_to_ans(rpn_to_tuple(rpn_string), var_dict)


def tuple_to_tree(tuple_representation: tuple) -> Node:
    """DEPRECATED -- convert tuple representation to expression tree"""
    *args, operator = tuple_representation
    args = [tuple_to_tree(arg) if isinstance(arg, tuple) else
            Variable(arg) if isinstance(arg, str) else
            Constant(arg) for arg in args]
    return operator_classes[operator](*args)


def rpn_to_tree(rpn_string: str) -> Node:
    """DEPRECATED -- convert rpn string to expression tree"""
    return tuple_to_tree(rpn_to_tuple(rpn_string))


def tuple_to_rpn(exp: Union[tuple, str, Number]) -> str:
    """convert tuple representation to rpn string"""
    out = ''
    if isinstance(exp, str):
        out += exp
    elif isinstance(exp, (int, float)):
        out += str(exp)
    elif isinstance(exp, tuple):
        for item in exp:
            out += f' {tuple_to_rpn(item)}'
    return out.strip()

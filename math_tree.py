"""
basic expression tree with evaluation and derivation
"""
from __future__ import annotations

import webbrowser
from abc import ABCMeta, abstractmethod
from fractions import Fraction
from functools import reduce
from math import acos, asin, atan, ceil, cos, e, factorial, floor, gamma, isclose, log, pi, sin, tan
from typing import Any, Optional, Union

ConstantType = Union[int, Fraction, float, complex, bool]
Environment = dict[str, ConstantType]


def tag(xml_tag: str, content: str, args: Optional[str] = None) -> str:
    """XML tag wrapping function"""
    if args is None:
        return f'<{xml_tag}>{content}</{xml_tag}>'
    else:
        return f'<{xml_tag} {args}>{content}</{xml_tag}>'


def mathml_tag(xml_tag: str, content: str, args: Optional[str] = None) -> str:
    """Mathml tag wrapping function"""
    return tag('m' + xml_tag, content, args)


def generate_html_code(expression: Node) -> str:
    """generates html code for expression"""
    return '<!DOCTYPE html>' \
           + tag('html',
                 tag('head',
                     tag('title',
                         'python_algebra output'))
                 + tag('body',
                       tag('math',
                           expression.mathml(),
                           'xmlns = "http://www.w3.org/1998/Math/MathML" id = "expr"')), 'lang=\'en\'')


def generate_html_doc(expression: Node) -> None:
    """generates html document for expression"""
    html = generate_html_code(expression)
    with open('output.html', 'w') as file:
        file.write(html)


def display(expression: Node) -> None:
    """Generates and opens html representation of expression"""
    generate_html_doc(expression)
    webbrowser.open('output.html')


def Nodeify(other: Union[Node, ConstantType, str]) -> Node:
    """turn given input into constant or variable leaf node"""
    if isinstance(other, Node):
        return other
    elif isinstance(other, bool):
        return Boolean(other)
    elif isinstance(other, int):
        return Integer(other)
    elif isinstance(other, Fraction):
        if other.denominator == 1:
            return Integer(int(other))
        elif other.denominator >= 1e6:
            return Real(float(other))
        else:
            return Rational(other)
    elif isinstance(other, float):
        if other.is_integer() and other <= 1e6:
            return Integer(int(other))
        elif other.as_integer_ratio()[1] < 1e6:
            return Rational(Fraction(other))
        else:
            return Real(other)
    elif isinstance(other, complex):
        return Complex(other)
    elif isinstance(other, str):
        return Variable(other)
    else:
        raise ValueError(f'Unsupported term type {type(other)} of {other}')


class EvaluationError(Exception):
    """Error raised if evaluation goes badly"""


class Node(metaclass=ABCMeta):
    """Abstract Base Class for any node in the expression tree"""
    __slots__ = '_finished',

    def __init__(self) -> None:
        self._finished = True

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return self.infix()

    def __bool__(self) -> bool:
        return bool(self.evaluate())

    def __float__(self) -> float:
        if isinstance(value := self.evaluate(), complex):
            return float(value.real)
        else:
            return float(value)

    def __int__(self) -> int:
        if isinstance(value := self.evaluate(), complex):
            return int(value.real)
        else:
            return int(value)

    def __complex__(self) -> complex:
        return complex(self.evaluate())

    def __eq__(self, other: Any) -> bool:
        return self is other or repr(self.simplify()) == repr(Nodeify(other).simplify())

    def __hash__(self) -> int:
        return hash(repr(self))

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __gt__(self, other: Any) -> bool:
        return GreaterThan(self, Nodeify(other)).evaluate()

    def __ge__(self, other: Any) -> bool:
        return GreaterEqual(self, Nodeify(other)).evaluate()

    def __lt__(self, other: Any) -> bool:
        return LessThan(self, Nodeify(other)).evaluate()

    def __le__(self, other: Any) -> bool:
        return LessEqual(self, Nodeify(other)).evaluate()

    def __add__(self, other: Union[Node, ConstantType, str]) -> Sum:
        try:
            return Sum(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __radd__(self, other: Union[Node, ConstantType, str]) -> Sum:
        try:
            return Sum(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __sub__(self, other: Union[Node, ConstantType, str]) -> Node:
        try:
            return Subtraction(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rsub__(self, other: Union[Node, ConstantType, str]) -> Node:
        try:
            return Subtraction(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __mul__(self, other: Union[Node, ConstantType, str]) -> Product:
        try:
            return Product(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rmul__(self, other: Union[Node, ConstantType, str]) -> Product:
        try:
            return Product(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __truediv__(self, other: Union[Node, ConstantType, str]) -> Node:
        try:
            return Division(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rtruediv__(self, other: Union[Node, ConstantType, str]) -> Node:
        try:
            return Division(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __pow__(self, other: Union[Node, ConstantType, str]) -> Exponent:
        try:
            return Exponent(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rpow__(self, other: Union[Node, ConstantType, str]) -> Exponent:
        try:
            return Exponent(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __neg__(self) -> Negate:
        return Negate(self)

    def __invert__(self) -> Not:
        return Not(self)

    def __and__(self, other: Union[Node, ConstantType, str]) -> And:
        try:
            return And(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rand__(self, other: Union[Node, ConstantType, str]) -> And:
        try:
            return And(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __or__(self, other: Union[Node, ConstantType, str]) -> Or:
        try:
            return Or(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __ror__(self, other: Union[Node, ConstantType, str]) -> Or:
        try:
            return Or(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __xor__(self, other: Union[Node, ConstantType, str]) -> Xor:
        try:
            return Xor(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rxor__(self, other: Union[Node, ConstantType, str]) -> Xor:
        try:
            return Xor(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __setattr__(self, key: str, value: Any) -> None:
        if not hasattr(self, '_finished') or not self._finished:
            super().__setattr__(key, value)
        else:
            raise AttributeError(f'\'{self.__class__.__name__}\' object is read-only')

    @abstractmethod
    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""

    @abstractmethod
    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""

    @abstractmethod
    def infix(self) -> str:
        """returns infix representation of the tree"""

    @abstractmethod
    def list_nodes(self) -> list[Node]:
        """return a list of all nodes in the tree"""

    @abstractmethod
    def mathml(self) -> str:
        """returns the MathML representation of the tree"""

    @abstractmethod
    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""

    @abstractmethod
    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""

    @abstractmethod
    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return set()

    def display(self) -> None:
        """shows graphical representation of expression"""
        display(self)

    def total_derivative(self) -> Node:
        """
        returns an expression tree representing the total derivative of this tree.
        the total derivative of f is defined as sum(f.derivative(var) for var in f.dependencies)
        """
        out = Sum(*(self.derivative(variable) for variable in self.dependencies()))
        return out


class Term(Node, metaclass=ABCMeta):
    """Abstract Base Class for any value (leaf node) in the expression tree"""
    __slots__ = ()

    def list_nodes(self) -> list[Node]:
        """returns a list of all nodes in the tree"""
        return [self]


class Constant(Term, metaclass=ABCMeta):
    """constant term in expression tree"""
    __slots__ = ()

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        return Nodeify(self.evaluate())


class Integer(Constant):
    """integer number in expression tree"""
    __slots__ = ('value',)

    def __init__(self, value: int) -> None:
        self.value = value
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.value)

    def evaluate(self, env: Optional[Environment] = None) -> int:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.value

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('n',
                                     str(self.value)))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return str(self.value)


class Rational(Constant):
    """rational number in expression tree"""
    __slots__ = ('value',)

    def __init__(self, value: Fraction) -> None:
        self.value = value
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.value)

    def evaluate(self, env: Optional[Environment] = None) -> Fraction:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.value

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('frac',
                                     mathml_tag('row',
                                                mathml_tag('n', str(self.value.numerator)))
                                     + mathml_tag('row',
                                                  mathml_tag('n', str(self.value.denominator)))))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'FractionBox[{self.value.numerator}, {self.value.denominator})'


class Real(Constant):
    """real number in expression tree"""
    __slots__ = ('value',)

    def __init__(self, value: float) -> None:
        self.value = value
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.value)

    def evaluate(self, env: Optional[Environment] = None) -> float:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.value

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('n',
                                     str(self.value)))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return str(self.value)


class Complex(Constant):
    """real number in expression tree"""
    __slots__ = ('value',)

    def __init__(self, value: complex) -> None:
        self.value = value
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'({self.value})'

    def evaluate(self, env: Optional[Environment] = None) -> complex:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.value

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('n',
                                     str(self.value.real))
                          + mathml_tag('o', '+')
                          + mathml_tag('row',
                                       mathml_tag('n', str(self.value.imag))
                                       + mathml_tag('i', 'i')))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.value.real} + {self.value.imag} \\[ImaginaryI]'


class Pi(Constant):
    """mathematical constant in expression tree"""
    __slots__ = ('value',)

    def __init__(self) -> None:
        self.value = pi
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return 'pi'

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.value

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('n', 'PI'))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return 'Pi'

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        return self


class E(Constant):
    """mathematical constant in expression tree"""
    __slots__ = ('value',)

    def __init__(self) -> None:
        self.value = e
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return 'e'

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.value

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('n', 'E'))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return 'E'

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        return self


class Boolean(Constant):
    """real number in expression tree"""
    __slots__ = ('value',)

    def __init__(self, value: bool) -> None:
        self.value = value
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.value)

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.value

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('i',
                                     str(self.value)))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return str(self.value)

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        return self


class Variable(Term):
    """Named variable in expression tree"""
    __slots__ = ('name',)

    def __init__(self, value: str) -> None:
        assert isinstance(value, str)
        self.name = value
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\'{self.name}\')'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.name)

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return {self.name}

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if self.name == variable:
            return Integer(1)
        return Integer(0)

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        if env is None:
            env = {}
        try:
            return Nodeify(env[self.name]).evaluate()
        except Exception as ex:
            raise EvaluationError from ex

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('i',
                                     str(self.name)))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        if env is None:
            env = {}
        if self.name in env.keys():
            return Nodeify(env[self.name])
        else:
            return self

    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        if self.name == var:
            return sub
        return self

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return self.name


class ArbitraryOperator(Node, metaclass=ABCMeta):
    """Abstract Base Class for multi-input operator in expression tree"""
    __slots__ = 'children',
    symbol = ''
    _parentheses_needed = '()'

    @property
    @abstractmethod
    def wolfram_func(self) -> str:
        """abstract property, returns function name for wolfram language"""

    @property
    @abstractmethod
    def default_value(self) -> ConstantType:
        """abstract property, returns function name for wolfram language"""

    def __init__(self, *args: Node) -> None:
        self.children = tuple(child for child in args)
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.children}'

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        # ugly but at least mypy shuts up
        return set('').union(*(child.dependencies() for child in self.children)).difference(set(''))

    @staticmethod
    @abstractmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        try:
            return reduce(self._eval_func, (child.evaluate(env) for child in self.children), self.default_value)
        except Exception as ex:
            raise EvaluationError from ex

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return self.symbol.join(child.infix() if not isinstance(child, eval(self._parentheses_needed))
                                else f"({child.infix()})" for child in self.children)

    def list_nodes(self) -> list[Node]:
        """returns a list of all nodes in the tree"""
        return sum((child.list_nodes() for child in self.children), [self])

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', self.symbol).join(child.mathml()
                                                            if isinstance(child, eval(self._parentheses_needed))
                                                            else mathml_tag('fenced', mathml_tag('row', child.mathml()))
                                                            for child in self.children))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        # try to return single constant
        try:
            return Nodeify(self.evaluate(env)).simplify()
        except EvaluationError:
            pass
        children = list(self.children)
        old_repr = ''
        while (new_repr := repr(children)) != old_repr:
            # update loop condition
            old_repr = new_repr
            # simplify the children
            children = [child.simplify(env) for child in children]
            # consolidate all constants
            children = self._consolidate_constants(children)
            # use operator specific rules
            children = self._simplify_terms(children, env)
            # sort children
            children.sort(key=lambda x: x.infix())
        children = [child.simplify(env) for child in children]
        # sort children
        children.sort(key=lambda x: x.infix())
        # figure out what to return
        if len(children) > 1:
            out = self.__class__(*children)
            try:
                return Nodeify(out.evaluate(env)).simplify()
            except EvaluationError:
                return out
        else:
            return children[0]

    def _consolidate_constants(self, children: list[Node]) -> list[Node]:
        """takes in a list of child nodes, consolidates all constants in the list"""
        constants: list[Node] = []
        non_constants: list[Node] = []
        for child in children:
            if isinstance(child, (Integer, Rational, Real, Complex)):
                constants.append(child)
            else:
                non_constants.append(child)
        if len(constants) > 1:
            children = non_constants + [self.__class__(*constants).simplify()]
        else:
            children = non_constants + constants
        return children

    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(*(child.substitute(var, sub) for child in self.children))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[' + ', '.join(child.wolfram() for child in self.children) + ']'

    @staticmethod
    @abstractmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """Simplification rules for operator"""


class Sum(ArbitraryOperator):
    """Addition operator node"""
    __slots__ = ()
    symbol = '+'
    wolfram_func = 'Plus'
    _parentheses_needed = '(ArbitraryLogicalOperator, ComparisonOperator)'
    default_value = 0

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Sum(*(child.derivative(variable) for child in self.children))

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""
        return x + y

    @staticmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """returns a simplified version of the tree"""

        def separate(arr: tuple[Node, ...]) -> tuple[Node, tuple[Node, ...]]:
            """separates array into a constant and any non-constant parts"""
            if any(isinstance(x, Constant) for x in arr):
                constant = next(filter(lambda x: isinstance(x, Constant), arr))
                non_constants = arr[:(k := arr.index(constant))] + arr[k + 1:]
            else:
                constant = Integer(1)
                non_constants = arr
            return constant, non_constants

        if len(children) == 1:
            return children
        elif len(children) == 0:
            return [Integer(0)]
        for i, child in enumerate(children):
            # eliminate zeroes
            if isinstance(child, Constant):
                if child.evaluate() == 0:
                    del children[i]
                    return children
            # consolidate sums
            elif isinstance(child, Sum):
                del children[i]
                return children + list(child.children)
            # eliminate negations
            elif isinstance(child, Negate):
                if child.child in children:
                    j = children.index(child.child)
                    del children[max(i, j)], children[min(i, j)]
                    if len(children) > 0:
                        return children
                    else:
                        return [Integer(0)]
                else:
                    for j, child2 in enumerate(children):
                        if isinstance(child2, Product):
                            child2_constant, child2_variable_terms = separate(child2.children)
                            if isinstance(child.child, Product):
                                if repr(child.child.children) == repr(child2_variable_terms):
                                    del children[max(i, j)], children[min(i, j)]
                                    return children + [Product(child2_constant - 1, *child.child.children)]
                                elif len(child2_variable_terms) == 1 and repr(child.child) == repr(
                                        child2_variable_terms[0]):
                                    del children[max(i, j)], children[min(i, j)]
                                    return children + [Product(child2_constant - 1, child.child)]
            # join like products
            elif isinstance(child, Product):
                constant1, non_constants1 = separate(child.children)
                for j, child2 in enumerate(children):
                    if i != j and isinstance(child2, Product):
                        child2_constant, child2_variable_terms = separate(child2.children)
                        if repr(non_constants1) == repr(child2_variable_terms):
                            del children[max(j, i)], children[min(j, i)]
                            return children + [Product(Sum(constant1, child2_constant), *non_constants1)]
            # assimilate like terms into products
            else:
                for j, child2 in enumerate(children):
                    if i == j:
                        pass
                    elif isinstance(child2, Product) and len(child2.children) == 2:
                        a, b = child2.children
                        if repr(a) == repr(child) and isinstance(b, Constant):
                            del children[max(i, j)], children[min(i, j)]
                            return children + [Product(b + 1, a).simplify()]
                        elif isinstance(a, Constant) and repr(b) == repr(child):
                            del children[max(i, j)], children[min(i, j)]
                            return children + [Product(a + 1, b).simplify()]
                    elif repr(child) == repr(child2):
                        del children[max(i, j)], children[min(i, j)]
                        return children + [Product(Integer(2), child).simplify()]
        return children


def Subtraction(*args: Node) -> Node:
    """Subtraction operator node"""
    return Sum(args[0], Negate(Sum(*args[1:]) if len(args) > 2 else args[1]))


class Product(ArbitraryOperator):
    """Multiplication operator node"""
    __slots__ = ()
    symbol = '*'
    wolfram_func = 'Times'
    _parentheses_needed = '(Sum, Modulus, ArbitraryLogicalOperator, ComparisonOperator)'
    default_value = 1

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if len(self.children) > 2:
            return Sum(Product(self.children[0], Product(*self.children[1:]).derivative(variable)),
                       Product(self.children[0].derivative(variable), *self.children))
        else:
            return Sum(Product(self.children[0], self.children[1].derivative(variable)),
                       Product(self.children[0].derivative(variable), self.children[1]))

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""
        return x * y

    @staticmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """returns a simplified version of the tree"""
        if len(children) == 1:
            return children
        elif len(children) == 0:
            return [Integer(0)]
        for i, child in enumerate(children):
            # eliminate ones and zeros
            if isinstance(child, Constant):
                if (ans := child.evaluate()) == 1:
                    del children[i]
                    return children
                elif ans == 0:
                    return [Integer(0)]
            # consolidate child products
            elif isinstance(child, Product):
                del children[i]
                return children + list(child.children)
            # distribute over sums
            elif isinstance(child, Sum):
                del children[i]
                return [Sum(*(Product(child2, *children) for child2 in child.children)).simplify(env)]
            # consolidate exponents
            elif isinstance(child, Exponent):
                for j, child2 in enumerate(children):
                    if i != j and isinstance(child2, Exponent) and repr(child.child1) == repr(child2.child1):
                        del children[j], children[i]
                        return children + [Exponent(child.child1, Sum(child.child2, child2.child2)).simplify(env)]
            # remove inversions
            elif isinstance(child, Invert):
                if child.child in children:
                    j = children.index(child.child)
                    del children[max(i, j)], children[min(i, j)]
                    if len(children) > 0:
                        return children
                    else:
                        return [Integer(0)]
                else:
                    for j, child2 in enumerate(children):
                        if isinstance(child2, Exponent) and repr(child2.child2) == repr(child):
                            del children[max(i, j)], children[min(i, j)]
                            return children + [Exponent(child, child2.child2 - 1)]
            # put like terms into exponents
            else:
                for j, child2 in enumerate(children):
                    if i == j:
                        pass
                    elif isinstance(child2, Exponent):
                        if repr(child) == repr(child2.child1):
                            del children[max(i, j)], children[min(i, j)]
                            return children + [Exponent(child, child2.child2 + 1).simplify()]
                    elif repr(child) == repr(child2):
                        del children[max(i, j)], children[min(i, j)]
                        return children + [Exponent(child, Integer(2)).simplify()]
        return children


def Division(*args: Node) -> Node:
    """Division operator node"""
    return Product(args[0], Invert(Product(*args[1:]) if len(args) > 2 else args[1]))


class Modulus(ArbitraryOperator):
    """Modulo operator node"""
    __slots__ = ()
    symbol = '%'
    wolfram_func = 'Mod'
    _parentheses_needed = '(ArbitraryOperator, Derivative)'
    default_value = 0

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        out = self.children[0].derivative(variable)
        for i, child in enumerate(self.children[1:]):
            out = Subtraction(out, Product(child.derivative(variable),
                                           Floor(Division(Modulus(*self.children[:i + 1], Integer(1)), child))))
        return out

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""
        if not (isinstance(x, complex) or isinstance(y, complex)):
            return x % y
        elif isinstance(x, complex) and not isinstance(y, complex):
            return x.real % y + x.imag % y
        else:
            raise NotImplementedError('mod of complex numbers not implemented')

    @staticmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """returns a simplified version of the tree"""
        return children


class BinaryOperator(Node, metaclass=ABCMeta):
    """Abstract Base Class for 2-input operator in expression tree"""
    __slots__ = 'child1', 'child2'
    wolfram_func = ''

    def __init__(self, child1: Node, child2: Node) -> None:
        self.child1 = child1
        self.child2 = child2
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child1)}, {repr(self.child2)})'

    def list_nodes(self) -> list[Node]:
        """return a list of all nodes in the tree"""
        out: list[Node] = []
        return out + [self] + self.child1.list_nodes() + self.child2.list_nodes()

    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child1.substitute(var, sub), self.child2.substitute(var, sub))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child1.wolfram()}, {self.child2.wolfram()}]'


class Exponent(BinaryOperator):
    """Exponent operator node"""
    __slots__ = ()
    symbol = '**'
    wolfram_func = 'Power'
    _parentheses_needed = '(ArbitraryOperator, Derivative)'

    def __init__(self, child1: Node, child2: Optional[Node] = None):
        if child2 is None:
            child1, child2 = E(), child1
        super().__init__(child1, child2)

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Product(self,
                       Sum(Product(self.child1.derivative(variable),
                                   Division(self.child2,
                                            self.child1)),
                           Product(self.child2.derivative(variable),
                                   Logarithm(self.child1,
                                             E()))))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluate expression tree"""
        try:
            ans1 = self.child1.evaluate(env)
            ans2 = self.child2.evaluate(env)
            test1 = float(ans1) if not isinstance(ans1, complex) else float(ans1.real ** 2 + ans1.imag ** 2) ** 0.5
            test2 = float(ans2) if not isinstance(ans2, complex) else float(ans2.real ** 2 + ans2.imag ** 2) ** 0.5
            float(test1) ** float(test2)
            return ans1 ** ans2
        except Exception as ex:
            raise EvaluationError from ex

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('sup',
                                     (self.child1.mathml() if isinstance(self.child1, eval(self._parentheses_needed))
                                      else mathml_tag('row',
                                                      mathml_tag('fenced',
                                                                 mathml_tag('row', self.child1.mathml()))))
                                     + (self.child2.mathml() if isinstance(self.child1, eval(self._parentheses_needed))
                                        else mathml_tag('row',
                                                        mathml_tag('fenced',
                                                                   mathml_tag('row', self.child2.mathml()))))))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return ((self.child1.infix() if isinstance(self.child1, (Term, UnaryOperator)) else f"({self.child1.infix()})")
                + self.symbol
                + (self.child2.infix() if isinstance(self.child2,
                                                     (Term, UnaryOperator)) else f"({self.child2.infix()})"))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify(env)
        child2 = self.child2.simplify(env)
        try:
            return Nodeify(self.__class__(child1, child2).evaluate(env)).simplify()
        except EvaluationError:
            pass
        # special cases for powers
        if isinstance(child2, Constant):
            if (ans2 := child2.evaluate()) == 0:
                return Integer(1)
            elif ans2 == 1:
                return child1
            elif ans2 == -1:
                return Invert(child1)
        # special cases for bases
        if isinstance(child1, Constant):
            if (ans1 := child1.evaluate()) == 0:
                return Integer(0)
            elif ans1 == 1:
                return Integer(1)
        # distribute over products
        elif isinstance(child1, Product):
            return Product(*(Exponent(child, child2) for child in child1.children)).simplify(env)
        # nested exponents multiply
        elif isinstance(child1, Exponent):
            return Exponent(child1.child1, Product(child1.child2, child2)).simplify(env)
        elif isinstance(child1, Sum) and isinstance(child2, Integer) and child2.evaluate() > 0:
            return Product(*([child1] * child2.evaluate())).simplify(env)
        return self.__class__(child1, child2)


class Logarithm(BinaryOperator):
    """Logarithm operator node, child 2 is base. default base is e"""
    __slots__ = ()
    symbol = 'log'
    wolfram_func = 'Log'
    _parentheses_needed = '()'

    def __init__(self, child1: Node, child2: Optional[Node] = None):
        if child2 is None:
            child2 = E()
        super().__init__(child1, child2)

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        x = self.child1.evaluate(env)
        y = self.child2.evaluate(env)
        if isinstance(x, complex):
            if x.imag == 0:
                x = float(x.real)
            else:
                raise EvaluationError from TypeError('log of complex number')
        if isinstance(y, complex):
            if y.imag == 0:
                y = float(y.real)
            else:
                raise EvaluationError from TypeError('log with complex base')
        try:
            return log(x, y)
        except Exception as ex:
            raise EvaluationError from ex

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(Subtraction(Division(Product(self.child1.derivative(variable),
                                                     Logarithm(self.child2,
                                                               E())),
                                             self.child1),
                                    Division(Product(self.child2.derivative(variable),
                                                     Logarithm(self.child1,
                                                               E())),
                                             self.child2)),
                        Exponent(Logarithm(self.child2,
                                           E()),
                                 Integer(2)))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'{self.symbol}({self.child1.infix()}, {self.child2.infix()})'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('sub',
                                     mathml_tag('i', self.symbol)
                                     + self.child2.mathml())
                          + mathml_tag('fenced', self.child1.mathml()))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child2.wolfram()}, {self.child1.wolfram()}]'

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify(env)
        child2 = self.child2.simplify(env)
        try:
            return Nodeify(self.__class__(child1, child2).evaluate(env)).simplify()
        except EvaluationError:
            pass
        if isinstance(child1, Product):
            return Sum(*(Logarithm(child, child2) for child in child1.children)).simplify(env)
        elif isinstance(child1, Exponent):
            return Product(child1.child2, Logarithm(child1.child1, child2)).simplify(env)
        return self.__class__(child1, child2)


class ArbitraryLogicalOperator(ArbitraryOperator, metaclass=ABCMeta):
    """Abstract base class for comparison operators"""
    __slots__ = ()
    _parentheses_needed = '(ArbitraryOperator, Derivative)'
    default_value = False

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)


class And(ArbitraryLogicalOperator):
    """logical AND operator node"""
    __slots__ = ()
    symbol = '&'
    wolfram_func = 'And'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return bool(x) & bool(y)

    @staticmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """returns a simplified version of the tree"""
        for i, child in enumerate(children):
            if isinstance(child, Constant):
                if child.evaluate():
                    del children[i]
                    return children if len(children) else [Boolean(True)]
                else:
                    return [Boolean(False)]
            elif isinstance(child, And):
                del children[i]
                return children + list(child.children)
            elif isinstance(child, Not):
                if child.child in children:
                    return [Boolean(False)]
        return children


class Or(ArbitraryLogicalOperator):
    """logical OR operator node"""
    __slots__ = ()
    symbol = '|'
    wolfram_func = 'Or'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return bool(x) | bool(y)

    @staticmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """returns a simplified version of the tree"""
        for i, child in enumerate(children):
            if isinstance(child, Constant):
                if not child.evaluate():
                    del children[i]
                    return children if len(children) else [Boolean(False)]
                else:
                    return [Boolean(True)]
            elif isinstance(child, Or):
                del children[i]
                return children + list(child.children)
            elif isinstance(child, Not):
                if child.child in children:
                    return [Boolean(True)]
        return children


class Xor(ArbitraryLogicalOperator):
    """logical XOR operator node"""
    __slots__ = ()
    symbol = '^'
    wolfram_func = 'Xor'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return bool(x) ^ bool(y)

    @staticmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """returns a simplified version of the tree"""
        if len(children) > 1:
            for i, child in enumerate(children):
                if isinstance(child, Constant):
                    if not child.evaluate():
                        del children[i]
                        return children if len(children) else [Boolean(False)]
                    else:
                        del children[i]
                        if len(children) > 1:
                            return [Not(Xor(*children)).simplify(env)]
                        elif len(children) == 1:
                            return [Not(children[0]).simplify(env)]
                        else:
                            return [Boolean(False)]
        return children


def Nand(*args: Node) -> Not:
    """logical NAND operator node"""
    return Not(And(*args))


def Nor(*args: Node) -> Not:
    """logical NOR operator node"""
    return Not(Or(*args))


def Xnor(*args: Node) -> Not:
    """logical XNOR operator node"""
    return Not(Xor(*args))


class ComparisonOperator(ArbitraryOperator, metaclass=ABCMeta):
    """Abstract base class for comparison operators"""
    __slots__ = ()
    _parentheses_needed = '(ComparisonOperator, )'

    def evaluate(self, env: Optional[Environment] = None) -> bool:
        """Evaluates the expression tree using the values from env, returns int or float"""
        try:
            return all(self._eval_func(x.evaluate(env), y.evaluate(env))
                       for x, y in zip(self.children[:-1], self.children[1:]))
        except Exception as ex:
            raise EvaluationError from ex

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    @staticmethod
    def _simplify_terms(children: list[Node], env: Optional[Environment] = None) -> list[Node]:
        """returns a simplified version of the tree"""
        return children


class IsEqual(ComparisonOperator):
    """Equality operator node"""
    __slots__ = ()
    symbol = '=='
    wolfram_func = 'EqualTo'
    default_value = True

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        if isinstance(x, (int, Fraction, float)) and isinstance(y, (int, Fraction, float)):
            return x == y or isclose(x, y)
        else:
            return x == y


def NotEqual(*args: Node) -> Node:
    """Inequality operator node"""
    return Not(IsEqual(*args))


class GreaterThan(ComparisonOperator):
    """Greater-than operator node"""
    __slots__ = ()
    symbol = '>'
    wolfram_func = 'Greater'
    default_value = False

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        if isinstance(x, complex) and x.imag == 0:
            x = x.real
        if isinstance(y, complex) and y.imag == 0:
            y = y.real
        if not (isinstance(x, complex) or isinstance(y, complex)):
            return x > y
        else:
            raise EvaluationError from TypeError('Comparison not defined in complex space')


class LessThan(ComparisonOperator):
    """Less-than operator node"""
    __slots__ = ()
    symbol = '<'
    wolfram_func = 'Less'
    default_value = False

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        if isinstance(x, complex) and x.imag == 0:
            x = x.real
        if isinstance(y, complex) and y.imag == 0:
            y = y.real
        if not (isinstance(x, complex) or isinstance(y, complex)):
            return x < y
        else:
            raise EvaluationError from TypeError('Comparison not defined in complex space')


class GreaterEqual(ComparisonOperator):
    """Greater-equal operator node"""
    __slots__ = ()
    symbol = '>='
    wolfram_func = 'GreaterEqual'
    default_value = True

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        if isinstance(x, complex) and x.imag == 0:
            x = x.real
        if isinstance(y, complex) and y.imag == 0:
            y = y.real
        if not (isinstance(x, complex) or isinstance(y, complex)):
            return x >= y
        else:
            raise EvaluationError from TypeError('Comparison not defined in complex space')


class LessEqual(ComparisonOperator):
    """Less-equal operator node"""
    __slots__ = ()
    symbol = '<='
    wolfram_func = 'LessEqual'
    default_value = True

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        """calculation function for 2 elements"""
        if isinstance(x, complex) and x.imag == 0:
            x = x.real
        if isinstance(y, complex) and y.imag == 0:
            y = y.real
        if not (isinstance(x, complex) or isinstance(y, complex)):
            return x <= y
        else:
            raise EvaluationError from TypeError('Comparison not defined in complex space')


class UnaryOperator(Node, metaclass=ABCMeta):
    """Abstract Base Class for single-input operator in expression tree"""
    __slots__ = 'child',
    symbol = ''
    wolfram_func = ''

    def __init__(self, child: Node) -> None:
        assert isinstance(child, Node)
        self.child = child
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child)})'

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies()

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'{self.symbol}({self.child.infix()})'

    def list_nodes(self) -> list[Node]:
        """returns a list of all nodes in the tree"""
        out: list[Node] = [self]
        return out + self.child.list_nodes()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('i', self.symbol)
                          + mathml_tag('fenced', self.child.mathml()))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        try:
            return Nodeify(self.evaluate(env)).simplify()
        except EvaluationError:
            pass
        new = self.__class__(self.child.simplify(env))
        try:
            return Nodeify(new.evaluate(env))
        except EvaluationError:
            pass
        return new

    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child.substitute(var, sub))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}]'


class Sine(UnaryOperator):
    """Sine operator node in radians"""
    __slots__ = ()
    symbol = 'sin'
    wolfram_func = 'Sin'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Product(Cosine(self.child),
                       self.child.derivative(variable))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child_ans = self.child.evaluate(env)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise EvaluationError from TypeError('mod of complex number')
        try:
            if (mod2pi := child_ans % 2 * pi) == 0 or mod2pi == pi:
                return 0
            elif mod2pi == pi / 2:
                return 1
            elif mod2pi == pi + pi / 2:
                return -1
            else:
                return sin(child_ans)
        except Exception as ex:
            raise EvaluationError from ex


class Cosine(UnaryOperator):
    """Cosine operator node in radians"""
    __slots__ = ()
    symbol = 'cos'
    wolfram_func = 'Cos'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Subtraction(Integer(0),
                           Product(Sine(self.child),
                                   self.child.derivative(variable)))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child_ans = self.child.evaluate(env)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise EvaluationError from TypeError('mod of complex number')
        try:
            if (mod2pi := child_ans % 2 * pi) == 0:
                return 1
            elif mod2pi == pi:
                return -1
            elif mod2pi == pi / 2 or mod2pi == pi + pi / 2:
                return 0
            else:
                return cos(child_ans)
        except Exception as ex:
            raise EvaluationError from ex


class Tangent(UnaryOperator):
    """Tangent operator node in radians"""
    __slots__ = ()
    symbol = 'tan'
    wolfram_func = 'Tan'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Cosine(self.child),
                                 Integer(2)))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child_ans = self.child.evaluate(env)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise EvaluationError from TypeError('mod of complex number')
        try:
            if (mod_pi := child_ans % pi) == 0:
                return 0
            elif mod_pi == pi / 2:
                raise EvaluationError from ValueError('tan of k*pi+pi/2 is infinity')
            else:
                return tan(child_ans)
        except Exception as ex:
            raise EvaluationError from ex


class ArcSine(UnaryOperator):
    """Arcsine operator node in radians"""
    __slots__ = ()
    symbol = 'asin'
    wolfram_func = 'ArcSin'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Subtraction(Integer(1),
                                             Exponent(self.child,
                                                      Integer(2))),
                                 Rational(Fraction(1 / 2))))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child_ans = self.child.evaluate(env)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise EvaluationError from TypeError('mod of complex number')
        if child_ans == 0:
            return 0
        elif child_ans == 1:
            return pi / 2
        else:
            try:
                return asin(child_ans)
            except Exception as ex:
                raise EvaluationError from ex


class ArcCosine(UnaryOperator):
    """Arccosine operator node in radians"""
    __slots__ = ()
    symbol = 'acos'
    wolfram_func = 'ArcCos'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Subtraction(Integer(0),
                           Division(self.child.derivative(variable),
                                    Exponent(Subtraction(Integer(1),
                                                         Exponent(self.child,
                                                                  Integer(2))),
                                             Rational(Fraction(1 / 2)))))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child_ans = self.child.evaluate(env)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise EvaluationError from TypeError('mod of complex number')
        if child_ans == 0:
            return pi / 2
        elif child_ans == 1:
            return 0
        elif child_ans == -1:
            return pi
        else:
            try:
                return acos(child_ans)
            except Exception as ex:
                raise EvaluationError from ex


class ArcTangent(UnaryOperator):
    """Arctangent operator node in radians"""
    __slots__ = ()
    symbol = 'atan'
    wolfram_func = 'ArcTan'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Sum(Integer(1),
                            Exponent(self.child,
                                     Integer(2))))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child_ans = self.child.evaluate(env)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise EvaluationError from TypeError('mod of complex number')
        if child_ans == 0:
            return 0
        else:
            try:
                return atan(child_ans)
            except Exception as ex:
                raise EvaluationError from ex


class Absolute(UnaryOperator):
    """Absolute operator node"""
    __slots__ = ()
    symbol = 'abs'
    wolfram_func = 'Abs'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(Product(self.child,
                                self.child.derivative(variable)),
                        Absolute(self.child))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child_ans = self.child.evaluate(env)
        try:
            if isinstance(child_ans, complex):
                return (child_ans.real ** 2 + child_ans.imag ** 2) ** 0.5
            elif child_ans >= 0:
                return child_ans
            else:
                return -child_ans
        except Exception as ex:
            raise EvaluationError from ex

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '|')
                          + self.child.mathml()
                          + mathml_tag('o', '|'))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        simplified = super().simplify(env)
        if isinstance(simplified, self.__class__):
            if isinstance(simplified.child, (Absolute, Negate)):
                return self.__class__(simplified.child.child).simplify(env)
            elif isinstance(simplified.child, Product):
                return Product(*(self.__class__(x) for x in simplified.child.children)).simplify(env)
        return simplified


class Negate(UnaryOperator):
    """Unary negative operator"""
    __slots__ = ()
    symbol = '-'
    wolfram_func = 'Minus'
    _parentheses_needed = '(ArbitraryOperator, Negate, Derivative, Piecewise, Factorial)'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Negate(self.child.derivative(variable))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        try:
            return -self.child.evaluate(env)
        except Exception as ex:
            raise EvaluationError from ex

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if isinstance(self.child, eval(self._parentheses_needed)):
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return mathml_tag('row',
                              mathml_tag('i', self.symbol)
                              + mathml_tag('fenced', self.child.mathml()))
        else:
            return mathml_tag('row',
                              mathml_tag('i', self.symbol)
                              + self.child.mathml())

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        simplified = super().simplify(env)
        if isinstance(simplified, self.__class__):
            if isinstance(simplified.child, Negate):
                return simplified.child.child.simplify(env)
            elif isinstance(simplified.child, Sum):
                return Sum(*(self.__class__(x) for x in simplified.child.children)).simplify(env)
        return simplified


class Invert(UnaryOperator):
    """Unary inversion operator"""
    __slots__ = ()
    symbol = '1/'
    wolfram_func = 'Divide'
    _parentheses_needed = '(ArbitraryOperator, Derivative, Piecewise, Factorial)'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if variable in self.dependencies():
            return Division(self.child.derivative(variable), Exponent(self.child, Integer(2)))
        else:
            return Integer(0)

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        child = self.child.evaluate(env)
        try:
            ans = 1 / child
            if isinstance(ans, complex):
                if ans.imag == 0:
                    ans = ans.real
                else:
                    return ans
            try:
                if int(ans) == ans:
                    final_ans: ConstantType = int(ans)
                else:
                    final_ans = ans
            except OverflowError:
                final_ans = ans
            return final_ans
        except Exception as ex:
            raise EvaluationError from ex

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if isinstance(self.child, eval(self._parentheses_needed)):
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('frac',
                                     mathml_tag('row',
                                                mathml_tag('n', '1'))
                                     + self.child.mathml()))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        simplified = super().simplify(env)
        if isinstance(simplified, self.__class__):
            if isinstance(simplified.child, Invert):
                return simplified.child.child.simplify()
        return simplified

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'Divide[1, {self.child.wolfram()}]'


class Floor(UnaryOperator):
    """floor operator"""
    __slots__ = ()
    symbol = 'floor'
    wolfram_func = 'Floor'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        ans = self.child.evaluate(env)
        try:
            return floor(ans) if not isinstance(ans, complex) else complex(floor(ans.real), floor(ans.imag))
        except Exception as ex:
            raise EvaluationError from ex

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '⌊')
                          + self.child.mathml()
                          + mathml_tag('o', '⌋'))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        simplified = super().simplify(env)
        if isinstance(simplified, self.__class__):
            if isinstance(simplified.child, (Floor, Ceiling)):
                return simplified.child.child.simplify(env)
        return simplified

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}]'


class Ceiling(UnaryOperator):
    """ceiling operator"""
    __slots__ = ()
    symbol = 'ceil'
    wolfram_func = 'Ceiling'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        ans = self.child.evaluate(env)
        try:
            return ceil(ans) if not isinstance(ans, complex) else complex(ceil(ans.real), ceil(ans.imag))
        except Exception as ex:
            raise EvaluationError from ex

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '⌈')
                          + self.child.mathml()
                          + mathml_tag('o', '⌉'))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        simplified = super().simplify(env)
        if isinstance(simplified, self.__class__):
            if isinstance(simplified.child, (Floor, Ceiling)):
                return simplified.child.child.simplify(env)
        return simplified

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}]'


class Factorial(UnaryOperator):
    """factorial operator"""
    __slots__ = ()
    symbol = '!'
    wolfram_func = 'Factorial'
    _parentheses_needed = '(ArbitraryOperator, Negate, Invert, Derivative, Piecewise)'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        raise NotImplementedError('derivative of factorial not implemented')

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        ans = self.child.evaluate(env)
        try:
            if isinstance(ans, int):
                return factorial(ans)
            elif isinstance(ans, complex) and ans.imag == 0 and ans.real % 1 == 0:
                return factorial(ans.real)
            elif not isinstance(ans, complex):
                return gamma(1 + ans)
            else:
                raise EvaluationError from TypeError('factorial not defined for complex number')
        except Exception as ex:
            raise EvaluationError from ex

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'({self.child.infix()}){self.symbol}'
        else:
            return f'{self.child.infix()}{self.symbol}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          self.child.mathml()
                          + mathml_tag('o', '!'))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'Factorial[{self.child.wolfram()}]'


class Not(UnaryOperator):
    """Logical not operator"""
    __slots__ = ()
    symbol = '~'
    wolfram_func = 'Not'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Not(self.child.derivative(variable))

    def evaluate(self, env: Optional[Environment] = None) -> bool:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return not self.child.evaluate(env)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return mathml_tag('row',
                              mathml_tag('i', self.symbol)
                              + mathml_tag('fenced', self.child.mathml()))
        else:
            return mathml_tag('row',
                              mathml_tag('i', self.symbol)
                              + self.child.mathml())

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        simplified = super().simplify(env)
        if isinstance(simplified, self.__class__):
            if isinstance(simplified.child, Not):
                return simplified.child.child.simplify(env)
            elif isinstance(simplified.child, And):
                return Or(*(Not(child2) for child2 in simplified.child.children)).simplify(env)
            elif isinstance(simplified.child, Or):
                return And(*(Not(child2) for child2 in simplified.child.children)).simplify(env)
        return simplified


class Derivative(Node):
    """Derivative operation node"""
    __slots__ = ('child', 'variable')
    wolfram_func = 'D'
    symbol = ''

    def __init__(self, expression: Node, variable: str) -> None:
        self.child = expression
        self.variable = variable
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child)}, {repr(self.variable)})'

    def list_nodes(self) -> list[Node]:
        """returns a list of all nodes in the tree"""
        out: list[Node] = [self]
        return out + self.child.list_nodes()

    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child.substitute(var, sub), self.variable)

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return self.child.derivative(self.variable).dependencies()

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Derivative(self, variable)

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        return self.child.derivative(self.variable).evaluate(env)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'd({self.child})/d{self.variable}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('frac',
                                     mathml_tag('row',
                                                mathml_tag('i',
                                                           'd'))
                                     + mathml_tag('row',
                                                  mathml_tag('i', 'd')
                                                  + mathml_tag('i', self.variable)))
                          + mathml_tag('fenced', self.child.mathml()))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""
        return self.child.simplify(env).derivative(self.variable).simplify(env)

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}, {self.variable}]'


class Piecewise(Node):
    """Piecewise function node"""
    wolfram_func = 'Piecewise'
    symbol = 'piecewise'
    __slots__ = 'expressions', 'default'

    def __init__(self, expressions: tuple[tuple[Node, Node], ...], default: Optional[Node] = None) -> None:
        self.default = default if default is not None else Integer(0)
        self.expressions = expressions
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.expressions)}, default={self.default})'

    def derivative(self, variable: str) -> Node:
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Piecewise(tuple((expr.derivative(variable), cond) for expr, cond in self.expressions),
                         self.default.derivative(variable))

    def evaluate(self, env: Optional[Environment] = None) -> ConstantType:
        """Evaluates the expression tree using the values from env, returns int or float"""
        for expression, condition in self.expressions:
            if condition.evaluate(env):
                return expression.evaluate(env)
        else:
            return self.default.evaluate(env)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        expression_part = ''
        for expr, cond in self.expressions:
            expression_part += f'({expr.infix()}, {cond.infix()}), '
        expression_part += self.default.infix()
        return self.symbol + '(' + expression_part + ')'

    def list_nodes(self) -> list[Node]:
        """returns a list of all nodes in the tree"""
        out: list[Node] = [self]
        for expr, cond in self.expressions:
            out += expr.list_nodes()
            out += cond.list_nodes()
        return out + self.default.list_nodes()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        expression_part = ''
        for expr, cond in self.expressions:
            expression_part += mathml_tag('tr',
                                          mathml_tag('td', expr.mathml())
                                          + mathml_tag('td',
                                                       mathml_tag('text',
                                                                  '&#xa0;<!--NO-BREAK SPACE-->'
                                                                  'if &#xa0;<!--NO-BREAK SPACE-->'),
                                                       'columnalign="left"')
                                          + mathml_tag('td', cond.mathml()))
        expression_part += mathml_tag('tr',
                                      mathml_tag('td', self.default.mathml())
                                      + mathml_tag('td',
                                                   mathml_tag('text', '&#xa0;<!--NO-BREAK SPACE--> otherwise'),
                                                   'columnalign="left"'))
        return mathml_tag('row',
                          mathml_tag('o', '{')
                          + mathml_tag('table',
                                       expression_part))

    def simplify(self, env: Optional[Environment] = None) -> Node:
        """returns a simplified version of the tree"""

        def check_child(child: Node) -> bool:
            """checks if node is always false"""
            try:
                return bool(child.evaluate(env))
            except (EvaluationError, TypeError):
                pass
            return True

        expressions = tuple(filter(lambda x: check_child(x[1]), self.expressions))
        return self.__class__(tuple((x.simplify(env), y.simplify(env)) for x, y in expressions),
                              self.default.simplify(env))

    def substitute(self, var: str, sub: Node) -> Node:
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return Piecewise(
            tuple((expr.substitute(var, sub), cond.substitute(var, sub)) for expr, cond in self.expressions),
            self.default.substitute(var, sub))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        expressions = ', '.join(f'{{{expr.wolfram()}, {cond.wolfram()}}}' for expr, cond in self.expressions)
        return f'{self.wolfram_func}[{{{expressions}}}, {self.default.wolfram()}]'

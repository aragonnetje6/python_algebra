"""
basic expression tree with evaluation and derivation
"""

import webbrowser
from abc import ABCMeta, abstractmethod
from fractions import Fraction
from functools import reduce
from math import pi, e, log, sin, cos, tan, asin, acos, atan, isclose, floor, ceil
from typing import Union, Optional, Any

ConstantType = Union[int, Fraction, float, complex, bool]
Variables = dict[str, ConstantType]


def tag(xml_tag: str, content: str, args: Optional[str] = None) -> str:
    """XML tag wrapping function"""
    if args is None:
        return f'<{xml_tag}>{content}</{xml_tag}>'
    else:
        return f'<{xml_tag} {args}>{content}</{xml_tag}>'


def mathml_tag(xml_tag: str, content: str, args: Optional[str] = None) -> str:
    """Mathml tag wrapping function"""
    return tag('m' + xml_tag, content, args)


def generate_html_code(expression: 'Node') -> str:
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


def generate_html_doc(expression: 'Node') -> None:
    """generates html document for expression"""
    html = generate_html_code(expression)
    with open('output.html', 'w') as file:
        file.write(html)


def display(expression: 'Node') -> None:
    """Generates and opens html representation of expression"""
    generate_html_doc(expression)
    webbrowser.open('output.html')


def Nodeify(other: Union['Node', ConstantType, str]) -> 'Node':
    """turn given input into constant or variable leaf node"""
    if isinstance(other, Node):
        return other.copy()
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
        if other.is_integer():
            return Integer(int(other))
        elif other.as_integer_ratio()[1] < 1e6:
            return Rational(Fraction(other))
        else:
            return Real(other)
    elif isinstance(other, complex):
        return Complex(other)
    elif isinstance(other, bool):
        return Boolean(other)
    elif isinstance(other, str):
        return Variable(other)
    else:
        raise ValueError(f'Unsupported term type {type(other)} of {other}')


class Node(metaclass=ABCMeta):
    """Abstract Base Class for any node in the expression tree"""
    __slots__ = 'parent', '_finished'

    def __init__(self) -> None:
        self.parent: Optional[Node] = None
        self.reset_parents()
        self._finished = True

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

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
        return IsEqual(self, Nodeify(other)).evaluate()

    def __ne__(self, other: Any) -> bool:
        return NotEqual(self, Nodeify(other)).evaluate()

    def __gt__(self, other: Any) -> bool:
        return GreaterThan(self, Nodeify(other)).evaluate()

    def __ge__(self, other: Any) -> bool:
        return GreaterEqual(self, Nodeify(other)).evaluate()

    def __lt__(self, other: Any) -> bool:
        return LessThan(self, Nodeify(other)).evaluate()

    def __le__(self, other: Any) -> bool:
        return LessEqual(self, Nodeify(other)).evaluate()

    def __add__(self, other: Union['Node', ConstantType, str]) -> 'Sum':
        try:
            return Sum(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __radd__(self, other: Union['Node', ConstantType, str]) -> 'Sum':
        try:
            return Sum(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __sub__(self, other: Union['Node', ConstantType, str]) -> 'Sum':
        try:
            return Subtraction(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rsub__(self, other: Union['Node', ConstantType, str]) -> 'Sum':
        try:
            return Subtraction(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __mul__(self, other: Union['Node', ConstantType, str]) -> 'Product':
        try:
            return Product(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rmul__(self, other: Union['Node', ConstantType, str]) -> 'Product':
        try:
            return Product(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __truediv__(self, other: Union['Node', ConstantType, str]) -> 'Product':
        try:
            return Division(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rtruediv__(self, other: Union['Node', ConstantType, str]) -> 'Product':
        try:
            return Division(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __pow__(self, other: Union['Node', ConstantType, str]) -> 'Exponent':
        try:
            return Exponent(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rpow__(self, other: Union['Node', ConstantType, str]) -> 'Exponent':
        try:
            return Exponent(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __neg__(self) -> 'Negate':
        return Negate(self)

    def __invert__(self) -> 'Not':
        return Not(self)

    def __and__(self, other: Union['Node', ConstantType, str]) -> 'And':
        try:
            return And(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rand__(self, other: Union['Node', ConstantType, str]) -> 'And':
        try:
            return And(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __or__(self, other: Union['Node', ConstantType, str]) -> 'Or':
        try:
            return Or(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __ror__(self, other: Union['Node', ConstantType, str]) -> 'Or':
        try:
            return Or(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __xor__(self, other: Union['Node', ConstantType, str]) -> 'Xor':
        try:
            return Xor(self, Nodeify(other))
        except ValueError:
            return NotImplemented

    def __rxor__(self, other: Union['Node', ConstantType, str]) -> 'Xor':
        try:
            return Xor(Nodeify(other), self)
        except ValueError:
            return NotImplemented

    def __setattr__(self, key: str, value: Any) -> None:
        if not hasattr(self, '_finished') or not self._finished or key == 'parent':
            super().__setattr__(key, value)
        else:
            raise AttributeError(f'\'{self.__class__.__name__}\' object is read-only')

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return set()

    @abstractmethod
    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""

    @abstractmethod
    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""

    @abstractmethod
    def infix(self) -> str:
        """returns infix representation of the tree"""

    @abstractmethod
    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""

    @abstractmethod
    def list_nodes(self) -> list['Node']:
        """return a list of all nodes in the tree"""

    @abstractmethod
    def mathml(self) -> str:
        """returns the MathML representation of the tree"""

    @abstractmethod
    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""

    @abstractmethod
    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""

    @abstractmethod
    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__()

    def display(self) -> None:
        """shows graphical representation of expression"""
        display(self)

    def get_root(self) -> 'Node':
        """Returns the root node of the expression tree"""
        if self.parent is None:
            return self
        return self.parent.get_root()

    def reset_parents(self, parent: Optional['Node'] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        self.parent = parent

    def total_derivative(self) -> 'Node':
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

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        return self.copy()


class Constant(Term, metaclass=ABCMeta):
    """constant term in expression tree"""
    __slots__ = ()

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.copy()


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

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var)).simplify()

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

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var)).simplify()

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

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var)).simplify()

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

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var)).simplify()

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

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var)).simplify()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('n', 'PI'))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return 'Pi'


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

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var)).simplify()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('n', 'E'))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return 'E'


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

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        raise TypeError('Integral of boolean expression')

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('i',
                                     str(self.value)))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return str(self.value)


class Variable(Term):
    """Named variable in expression tree"""
    __slots__ = ('name',)

    def __init__(self, value: str) -> None:
        assert isinstance(value, str)
        self.name = value
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\'{self.name}\')'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.name)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.name)

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return {self.name}

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if self.name == variable:
            return Integer(1)
        return Integer(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        if var_dict is None:
            raise KeyError(f'None does not contain "{self.name}"')
        return Nodeify(var_dict[self.name]).evaluate()

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if self.name == var:
            if self.parent is None or isinstance(self.parent, Sum):
                return Division(Exponent(self, Integer(2)), Integer(2)).simplify()
            else:
                return self.copy().simplify()
        else:
            return Integer(0)

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('i',
                                     str(self.name)))

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        if self.name == var:
            return sub.copy()
        return self.copy()

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return self.name


class ArbitraryOperator(Node, metaclass=ABCMeta):
    """Abstract Base Class for multi-input operator in expression tree"""
    __slots__ = 'children',
    symbol = ''

    @property
    @abstractmethod
    def wolfram_func(self) -> str:
        """abstract property, returns function name for wolfram language"""

    def __init__(self, *args: Node) -> None:
        assert len(args) > 1
        assert all(isinstance(x, Node) for x in args)
        self.children = tuple(child.copy() for child in args)
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.children}'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(*(child.copy() for child in self.children))

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        # ugly but at least mypy shuts up
        return set('').union(*(child.dependencies() for child in self.children)).difference(set(''))

    @staticmethod
    @abstractmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return reduce(self._eval_func, (child.evaluate(var_dict) for child in self.children))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if isinstance(self.parent, Invert) or isinstance(self.parent, Exponent):
            return '(' + self.symbol.join(child.infix() for child in self.children) + ')'
        else:
            return self.symbol.join(child.infix() for child in self.children)

    def list_nodes(self) -> list[Node]:
        """returns a list of all nodes in the tree"""
        return sum((child.list_nodes() for child in self.children), [self])

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if isinstance(self.parent, Invert) or isinstance(self.parent, Exponent):
            return mathml_tag('row',
                              mathml_tag('fenced',
                                         mathml_tag('row',
                                                    mathml_tag('o', self.symbol).join(
                                                        child.mathml() for child in self.children))))
        else:
            return mathml_tag('row',
                              mathml_tag('o', self.symbol).join(child.mathml() for child in self.children))

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        for child in self.children:
            child.reset_parents(self)

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        try:
            return Nodeify(self.evaluate())
        except (KeyError, ValueError):
            return self.__class__(*(child.simplify() for child in self.children))

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(*(child.substitute(var, sub) for child in self.children))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[' + ', '.join(child.wolfram() for child in self.children) + ']'


class Sum(ArbitraryOperator):
    """Addition operator node"""
    __slots__ = ()
    symbol = '+'
    wolfram_func = 'Plus'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Sum(*(child.derivative(variable) for child in self.children))

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""
        return x + y

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Sum, Logarithm, UnaryOperator)):
            return self.symbol.join(child.infix() for child in self.children)
        else:
            return '(' + self.symbol.join(child.infix() for child in self.children) + ')'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Sum(*(child.integral(var) for child in self.children)).simplify()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Sum, Logarithm, UnaryOperator)):
            return mathml_tag('row',
                              mathml_tag('o', self.symbol).join(child.mathml() for child in self.children))
        else:
            return mathml_tag('row',
                              mathml_tag('fenced',
                                         mathml_tag('row',
                                                    mathml_tag('o', self.symbol).join(
                                                        child.mathml() for child in self.children))))

    # todo: reimplement Sum.simplify
    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        children = [child.simplify() for child in self.children]
        # # evaluate full expression
        # try:
        #     return Constant(self.evaluate())
        # except KeyError:
        #     pass
        # # return only child
        # if len(children) == 1:
        #     return children[0]
        # while True:
        #     # flatten nested sum nodes
        #     if sums := [child for child in children if isinstance(child, Sum)]:
        #         children = [child for child in children if child not in sums]
        #         for sum_node in sums:
        #             children += sum_node.children
        #         continue
        #     # consolidate Negate nodes
        #     if len(negations := [child for child in children if isinstance(child, Negate)]) > 1:
        #         children = [child for child in children if child not in negations] + \
        #                    [Negate(*(negate.child for negate in negations)).simplify()]
        #         continue
        #     # consolidate constants
        #     if len(constants := [child for child in children if isinstance(child, Constant)]) > 1:
        #         total = Sum(*constants).evaluate()
        #         if total == 0:
        #             children = [child for child in children if child not in constants] + [Constant(total)]
        #         else:
        #             children = [child for child in children if child not in constants]
        #         continue
        #     # consolidate variables into products
        #     if len(variables := [child for child in children if isinstance(child, Variable)]) > 1:
        #         pass
        #         # separate variables
        #     # consolidate products
        #         # consolidate exponents of variables
        #     break
        return Sum(*children)


def Subtraction(*args: Node) -> Sum:
    """Subtraction operator node"""
    return Sum(args[0], Negate(*args[1:]))


class Product(ArbitraryOperator):
    """Multiplication operator node"""
    __slots__ = ()
    symbol = '*'
    wolfram_func = 'Times'

    def derivative(self, variable: str) -> 'Node':
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

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif len(tup := tuple(filter(lambda x: var in x.dependencies(), self.children))) == 1:
            return Product(*filter(lambda x: var not in x.dependencies(), self.children),
                           tup[0].integral(var)).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    # todo: reimplement Product.simplify
    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        children = [child.simplify() for child in self.children]
        # # evaluate full expression
        # try:
        #     return Constant(self.evaluate())
        # except KeyError:
        #     pass
        # # return only child
        # if len(children) == 1:
        #     return children[0]
        # for i, child in enumerate(children):
        #     # flatten nested product nodes
        #     if isinstance(child, Product):
        #         return Product(*children[:i], *children[i + 1:], *child.children).simplify()
        #     # consolidate Invert nodes
        #     elif isinstance(child, Invert):
        #         for j, child2 in enumerate(children[i + 1:]):
        #             if isinstance(child2, Invert):
        #                 return Product(*children[:i], *children[i + 1:j], *children[j + 1:],
        #                                Invert(Product(child.child, child2.child))).simplify()
        #     # distribute over sums
        #     elif isinstance(child, Sum):
        #         return Sum(*(Product(sub_child, *children[:i], *children[i + 1:]) for sub_child in
        #                      child.children)).simplify()
        #     elif isinstance(child, Constant):
        #         # eliminate multiplying by one
        #         if child.evaluate() == 1:
        #             return Product(*children[:i], *children[i + 1:]).simplify()
        #         # return zero if a term equals zero
        #         elif child.evaluate() == 0:
        #             return Integer(0)
        #         # attempt to consolidate constants
        #         else:
        #             for j, child2 in enumerate(children[i + 1:]):
        #                 if isinstance(child2, Constant):
        #                     return Product(Constant(Product(child, child2).evaluate()),
        #                                    *children[:i], *children[i + 1:j], *children[j + 1:]).simplify()
        #     # consolidate variables into exponents
        #     elif isinstance(child, Variable):
        #         for j, child2 in enumerate(children[i + 1:]):
        #             if isinstance(child2, Variable) and child.value == child2.value:
        #                 return Product(Exponent(child, Integer(2)), *children[:i], *children[i + 1:j],
        #                                *children[j + 1:]).simplify()
        #             elif isinstance(child2, Exponent) and isinstance(child2.child1, Variable) and \
        #                     child2.child1.value == child.value:
        #                 return Product(Exponent(child, Sum(child2.child2, Integer(1))), *children[:i],
        #                                *children[i + 1:j], *children[j + 1:]).simplify()
        #     # consolidate exponents
        #     elif isinstance(child, Exponent):
        #         # consolidate exponents of variables
        #         if isinstance(child.child1, Variable):
        #             for j, child2 in enumerate(children[i + 1:]):
        #                 if isinstance(child2, Variable) and child2.value == child.child1.value:
        #                     return Product(Exponent(child2, Sum(child.child2, Integer(1))), *children[:i],
        #                                    *children[i + 1:j], *children[j + 1:]).simplify()
        #                 elif isinstance(child2, Exponent) and isinstance(child2.child1, Variable) and \
        #                         child2.child1.value == child.child1.value:
        #                     return Product(Exponent(child.child1, Sum(child.child2, child2.child2)), *children[:i],
        #                                    *children[i + 1:j], *children[j + 1:]).simplify()
        return Product(*children)


def Division(*args: Node) -> Product:
    """Division operator node"""
    return Product(args[0], Invert(*args[1:]))


class Modulus(ArbitraryOperator):
    """Modulo operator node"""
    __slots__ = ()
    symbol = '*'
    wolfram_func = 'Mod'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        out = self.children[0].derivative(variable)
        for i, child in enumerate(self.children[1:]):
            out = Subtraction(out, Product(child.derivative(variable),
                                           Floor(Division(Modulus(*self.children[:i + 1], Integer(1)), child))))
        return out.simplify()

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""
        if not (isinstance(x, complex) or isinstance(y, complex)):
            return x % y
        elif isinstance(x, complex) and not isinstance(y, complex):
            return x.real % y + x.imag % y

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        raise NotImplementedError('Integration not supported for this expression')

    # todo: implement Modulus.simplify


class BinaryOperator(ArbitraryOperator, metaclass=ABCMeta):
    """Abstract Base Class for 2-input operator in expression tree"""
    __slots__ = 'child1', 'child2'

    def __init__(self, *args: Node):
        assert len(args) == 2
        self.child1 = args[0]
        self.child2 = args[1]
        super().__init__(*args)


class Exponent(BinaryOperator):
    """Exponent operator node"""
    __slots__ = ()
    symbol = '**'
    wolfram_func = 'Power'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Product(self,
                       Sum(Product(self.child1.derivative(variable),
                                   Division(self.child2,
                                            self.child1)),
                           Product(self.child2.derivative(variable),
                                   Logarithm(self.child1,
                                             E()))))

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""
        return x ** y

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if isinstance(self.parent, Exponent):
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif var not in self.child1.dependencies() and self.child2 == Variable(var):
            return Division(Exponent(self.child1,
                                     Variable(var)),
                            Logarithm(self.child1,
                                      E())).simplify()
        elif var not in self.child2.dependencies() and self.child1 == Variable(var):
            return Division(Exponent(Variable(var),
                                     Sum(self.child2,
                                         Integer(1))),
                            Sum(self.child2,
                                Integer(1))).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if isinstance(self.parent, Exponent):
            return mathml_tag('row',
                              mathml_tag('fenced',
                                         mathml_tag('row',
                                                    mathml_tag('sup',
                                                               self.child1.mathml()
                                                               + self.child2.mathml()))))
        else:
            return mathml_tag('row',
                              mathml_tag('sup',
                                         self.child1.mathml()
                                         + self.child2.mathml()))

    # todo: reimplement Exponent.simplify


class Logarithm(BinaryOperator):
    """Logarithm operator node, child 2 is base. default base is e"""
    __slots__ = ()
    symbol = 'log'
    wolfram_func = 'Log'

    def __init__(self, child1: Node, child2: Optional[Node] = None):
        if child2 is None:
            child2 = E()
        super().__init__(child1, child2)

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> ConstantType:
        """calculation function for 2 elements"""
        if isinstance(x, complex) or isinstance(y, complex):
            raise NotImplementedError('complex values of logarithms not supported')
        return log(x, y)

    def derivative(self, variable: str) -> 'Node':
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

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child1 == Variable(var) and len(self.child2.dependencies()) == 0 and self.child2.evaluate() == e:
            return Subtraction(Product(Variable(var),
                                       Logarithm(Variable(var),
                                                 E())),
                               Variable(var)).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

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


class ArbitraryLogicalOperator(ArbitraryOperator, metaclass=ABCMeta):
    """Abstract base class for comparison operators"""
    __slots__ = ()

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return '(' + self.symbol.join(child.infix() for child in self.children) + ')'
        else:
            return self.symbol.join(child.infix() for child in self.children)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        raise TypeError('Integral of boolean expression')


class And(ArbitraryLogicalOperator):
    """logical AND operator node"""
    __slots__ = ()
    symbol = '&'
    wolfram_func = 'And'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return bool(x) & bool(y)

    # todo: reimplement And.simplify
    # def simplify(self) -> Node:
    #     """returns a simplified version of the tree"""
    #     try:
    #         return Constant(self.evaluate())
    #     except KeyError:
    #         pass
    #     simple_child1 = self.child1.simplify()
    #     simple_child2 = self.child2.simplify()
    #     if isinstance(simple_child1, Not) and isinstance(simple_child2, Not):
    #         return Nor(simple_child1.child, simple_child2.child).simplify()
    #     else:
    #         return And(simple_child1, simple_child2)


class Or(ArbitraryLogicalOperator):
    """logical OR operator node"""
    __slots__ = ()
    symbol = '|'
    wolfram_func = 'Or'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return bool(x) | bool(y)

    # todo: reimplement Or.simplify
    # def simplify(self) -> Node:
    #     """returns a simplified version of the tree"""
    #     try:
    #         return Constant(self.evaluate())
    #     except KeyError:
    #         pass
    #     simple_child1 = self.child1.simplify()
    #     simple_child2 = self.child2.simplify()
    #     if isinstance(simple_child1, Not) and isinstance(simple_child2, Not):
    #         return Nand(simple_child1.child, simple_child2.child).simplify()
    #     else:
    #         return Nor(simple_child1, simple_child2)


class Xor(ArbitraryLogicalOperator):
    """logical XOR operator node"""
    __slots__ = ()
    symbol = '^'
    wolfram_func = 'Xor'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return bool(x) ^ bool(y)

    # todo: reimplement Xor.simplify
    # def simplify(self) -> Node:
    #     """returns a simplified version of the tree"""
    #     try:
    #         return Constant(self.evaluate())
    #     except KeyError:
    #         pass
    #     simple_child1 = self.child1.simplify()
    #     simple_child2 = self.child2.simplify()
    #     if isinstance(simple_child1, Not) and isinstance(simple_child2, Not):
    #         return Xor(simple_child1.child, simple_child2.child).simplify()
    #     else:
    #         return Xor(simple_child1, simple_child2)


class Nand(ArbitraryLogicalOperator):
    """logical NAND operator node"""
    __slots__ = ()
    wolfram_func = 'Nand'
    symbol = '&'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return not (bool(x) & bool(y))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return '(not ' + super().infix() + ')'
        else:
            return 'not ' + super().infix()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '~')
                          + mathml_tag('fenced',
                                       mathml_tag('row',
                                                  mathml_tag('o', '&').join(
                                                      child.mathml() for child in self.children))))

    # todo: reimplement Nand.simplify
    # def simplify(self) -> Node:
    #     """returns a simplified version of the tree"""
    #     try:
    #         return Constant(self.evaluate())
    #     except KeyError:
    #         pass
    #     simple_child1 = self.child1.simplify()
    #     simple_child2 = self.child2.simplify()
    #     if isinstance(simple_child1, Not) and isinstance(simple_child2, Not):
    #         return Or(simple_child1.child, simple_child2.child).simplify()
    #     else:
    #         return Nand(simple_child1, simple_child2)


class Nor(ArbitraryLogicalOperator):
    """logical NOR operator node"""
    __slots__ = ()
    wolfram_func = 'Nor'
    symbol = '|'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return not (bool(x) | bool(y))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return '(not ' + super().infix() + ')'
        else:
            return 'not ' + super().infix()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '~')
                          + mathml_tag('fenced',
                                       mathml_tag('row',
                                                  mathml_tag('o', '|').join(
                                                      child.mathml() for child in self.children))))

    # todo: reimplement Nor.simplify
    # def simplify(self) -> Node:
    #     """returns a simplified version of the tree"""
    #     try:
    #         return Constant(self.evaluate())
    #     except KeyError:
    #         pass
    #     simple_child1 = self.child1.simplify()
    #     simple_child2 = self.child2.simplify()
    #     if isinstance(simple_child1, Not) and isinstance(simple_child2, Not):
    #         return And(simple_child1.child, simple_child2.child).simplify()
    #     else:
    #         return Nor(simple_child1, simple_child2)


class Xnor(ArbitraryLogicalOperator):
    """logical XOR operator node"""
    __slots__ = ()
    wolfram_func = 'Xnor'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return not (bool(x) ^ bool(y))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return '(not ' + super().infix() + ')'
        else:
            return 'not ' + super().infix()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '~')
                          + mathml_tag('fenced',
                                       mathml_tag('row',
                                                  mathml_tag('o', '^').join(
                                                      child.mathml() for child in self.children))))

    # todo: reimplement Xnor.simplify
    # def simplify(self) -> Node:
    #     """returns a simplified version of the tree"""
    #     try:
    #         return Constant(self.evaluate())
    #     except KeyError:
    #         pass
    #     simple_child1 = self.child1.simplify()
    #     simple_child2 = self.child2.simplify()
    #     if isinstance(simple_child1, Not) and isinstance(simple_child2, Not):
    #         return Xnor(simple_child1.child, simple_child2.child).simplify()
    #     else:
    #         return Xnor(simple_child1, simple_child2)


class ComparisonOperator(ArbitraryOperator, metaclass=ABCMeta):
    """Abstract base class for comparison operators"""
    __slots__ = ()

    def evaluate(self, var_dict: Optional[Variables] = None) -> bool:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return all(self._eval_func(x.evaluate(var_dict), y.evaluate(var_dict))
                   for x, y in zip(self.children[:-1], self.children[1:]))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        raise ArithmeticError("Integration of logical operators not supported")


class IsEqual(ComparisonOperator):
    """Equality operator node"""
    __slots__ = ()
    symbol = '=='
    wolfram_func = 'EqualTo'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        if isinstance(x, complex) and x.imag == 0:
            x = x.real
        if isinstance(y, complex) and y.imag == 0:
            y = y.real
        if isinstance(x, bool) or isinstance(y, bool):
            return x == y
        elif not (isinstance(x, (complex, bool)) or isinstance(y, (complex, bool))):
            return x == y or isclose(x, y)
        else:
            raise TypeError('Comparison not defined in complex space')


class NotEqual(ComparisonOperator):
    """Inequality operator node"""
    __slots__ = ()
    symbol = '!='
    wolfram_func = 'UnequalTo'

    @staticmethod
    def _eval_func(x: ConstantType, y: ConstantType) -> bool:
        """calculation function for 2 elements"""
        return x != y


class GreaterThan(ComparisonOperator):
    """Greater-than operator node"""
    __slots__ = ()
    symbol = '>'
    wolfram_func = 'Greater'

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
            raise TypeError('Comparison not defined in complex space')


class LessThan(ComparisonOperator):
    """Less-than operator node"""
    __slots__ = ()
    symbol = '<'
    wolfram_func = 'Less'

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
            raise TypeError('Comparison not defined in complex space')


class GreaterEqual(ComparisonOperator):
    """Greater-equal operator node"""
    __slots__ = ()
    symbol = '>='
    wolfram_func = 'GreaterEqual'

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
            raise TypeError('Comparison not defined in complex space')


class LessEqual(ComparisonOperator):
    """Less-equal operator node"""
    __slots__ = ()
    symbol = '<='
    wolfram_func = 'LessEqual'

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
            raise TypeError('Comparison not defined in complex space')


class UnaryOperator(Node, metaclass=ABCMeta):
    """Abstract Base Class for single-input operator in expression tree"""
    __slots__ = 'child',
    symbol = ''
    wolfram_func = ''

    def __init__(self, child: Node) -> None:
        assert isinstance(child, Node)
        self.child = child.copy()
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child)})'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.child)

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies()

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'{self.symbol}({self.child.infix()})'

    def list_nodes(self) -> list[Node]:
        """returns a list of all nodes in the tree"""
        out = [self]  # type: list[Node]
        return out + self.child.list_nodes()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('i', self.symbol)
                          + mathml_tag('fenced', self.child.mathml()))

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        self.child.reset_parents(self)

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        try:
            return Nodeify(self.evaluate())
        except (KeyError, ValueError):
            return self.__class__(self.child.simplify())

    def substitute(self, var: str, sub: 'Node') -> 'Node':
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

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Product(Cosine(self.child),
                       self.child.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child_ans = self.child.evaluate(var_dict)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise TypeError('Sine of complex number')
        if (mod2pi := child_ans % 2 * pi) == 0 or mod2pi == pi:
            return 0
        elif mod2pi == pi / 2:
            return 1
        elif mod2pi == pi + pi / 2:
            return -1
        else:
            return sin(child_ans)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child == Variable(var):
            return Product(Integer(-1),
                           Cosine(Variable(var))).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    # todo: implement Sine.simplify


class Cosine(UnaryOperator):
    """Cosine operator node in radians"""
    __slots__ = ()
    symbol = 'cos'
    wolfram_func = 'Cos'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Subtraction(Integer(0),
                           Product(Sine(self.child),
                                   self.child.derivative(variable)))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child_ans = self.child.evaluate(var_dict)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise TypeError('Cosine of complex number')
        if (mod2pi := child_ans % 2 * pi) == 0:
            return 1
        elif mod2pi == pi:
            return -1
        elif mod2pi == pi / 2 or mod2pi == pi + pi / 2:
            return 0
        else:
            return cos(child_ans)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child == Variable(var):
            return Sine(Variable(var)).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    # todo: implement Cosine.simplify


class Tangent(UnaryOperator):
    """Tangent operator node in radians"""
    __slots__ = ()
    symbol = 'tan'
    wolfram_func = 'Tan'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Cosine(self.child),
                                 Integer(2)))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child_ans = self.child.evaluate(var_dict)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise TypeError('Tangent of complex number')
        if (mod_pi := child_ans % pi) == 0:
            return 0
        elif mod_pi == pi / 2:
            raise ValueError('tan of k*pi+pi/2 is infinity')
        else:
            return tan(child_ans)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child == Variable(var):
            return Product(Integer(-1),
                           Logarithm(Cosine(Variable(var)),
                                     E())).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    # todo: implement Tangent.simplify


class ArcSine(UnaryOperator):
    """Arcsine operator node in radians"""
    __slots__ = ()
    symbol = 'asin'
    wolfram_func = 'ArcSin'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Subtraction(Integer(1),
                                             Exponent(self.child,
                                                      Integer(2))),
                                 Rational(Fraction(1 / 2))))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child_ans = self.child.evaluate(var_dict)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise TypeError('ArcSine of complex number')
        if child_ans == 0:
            return 0
        elif child_ans == 1:
            return pi / 2
        else:
            return asin(child_ans)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child == Variable(var):
            return Sum(Product(Variable(var),
                               self),
                       Exponent(Subtraction(Integer(1),
                                            Exponent(Variable(var),
                                                     Integer(2))),
                                Rational(Fraction(1 / 2)))).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    # todo: implement ArcSine.simplify


class ArcCosine(UnaryOperator):
    """Arccosine operator node in radians"""
    __slots__ = ()
    symbol = 'acos'
    wolfram_func = 'ArcCos'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Subtraction(Integer(0),
                           Division(self.child.derivative(variable),
                                    Exponent(Subtraction(Integer(1),
                                                         Exponent(self.child,
                                                                  Integer(2))),
                                             Rational(Fraction(1 / 2)))))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child_ans = self.child.evaluate(var_dict)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise TypeError('ArcCosine of complex number')
        if child_ans == 0:
            return pi / 2
        elif child_ans == 1:
            return 0
        elif child_ans == -1:
            return pi
        else:
            return acos(child_ans)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child == Variable(var):
            return Subtraction(Product(Variable(var),
                                       self),
                               Exponent(Subtraction(Integer(1),
                                                    Exponent(Variable(var),
                                                             Integer(2))),
                                        Rational(Fraction(1 / 2)))).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    # todo: implement ArcCosine.simplify


class ArcTangent(UnaryOperator):
    """Arctangent operator node in radians"""
    __slots__ = ()
    symbol = 'atan'
    wolfram_func = 'ArcTan'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Sum(Integer(1),
                            Exponent(self.child,
                                     Integer(2))))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child_ans = self.child.evaluate(var_dict)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                raise TypeError('ArcTangent of complex number')
        if child_ans == 0:
            return 0
        else:
            return atan(child_ans)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child == Variable(var):
            return Sum(Product(Variable(var),
                               self),
                       Product(Rational(Fraction(1 / 2)),
                               Logarithm(Sum(Exponent(Variable(var),
                                                      Integer(2)),
                                             Integer(1)),
                                         E()))).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    # todo: implement ArcTangent.simplify


class Absolute(UnaryOperator):
    """Absolute operator node"""
    __slots__ = ()
    symbol = 'abs'
    wolfram_func = 'Abs'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(Product(self.child,
                                self.child.derivative(variable)),
                        Absolute(self.child))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child_ans = self.child.evaluate(var_dict)
        if isinstance(child_ans, complex):
            if child_ans.imag == 0:
                child_ans = child_ans.real
            else:
                return (child_ans.real ** 2 + child_ans.imag ** 2) ** 0.5
        if child_ans >= 0:
            return child_ans
        else:
            return -child_ans

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var)).simplify()
        elif self.child == Variable(var):
            return Division(Product(Variable(var),
                                    Absolute(Variable(var))),
                            Integer(2)).simplify()
        else:
            raise NotImplementedError('Integration not supported for this expression')

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '|')
                          + self.child.mathml()
                          + mathml_tag('o', '|'))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        try:
            return Nodeify(self.evaluate())
        except KeyError:
            pass
        child = self.child.simplify()
        if isinstance(child, (Absolute, Negate)):
            return Absolute(child.child)
        return Absolute(child)


class Negate(UnaryOperator):
    """Unary negative operator"""
    __slots__ = ()
    symbol = '-'
    wolfram_func = 'Minus'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Negate(self.child.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return -self.child.evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Negate(self.child.integral(var)).simplify()

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

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simple_child = self.child.simplify()
        try:
            return Nodeify(-simple_child.evaluate())
        except KeyError:
            pass
        if isinstance(simple_child, Negate):
            return simple_child.child
        else:
            return Negate(simple_child)


class Invert(UnaryOperator):
    """Unary inversion operator"""
    __slots__ = ()
    symbol = '1/'
    wolfram_func = 'Divide'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if variable in self.dependencies():
            return Division(self.child.derivative(variable), Exponent(self.child, Integer(2)))
        else:
            return Integer(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        ans = 1 / self.child.evaluate(var_dict)
        if isinstance(ans, complex):
            if ans.imag == 0:
                ans = ans.real
            else:
                return ans
        try:
            if int(ans) == ans:
                final_ans = int(ans)  # type: ConstantType
            else:
                final_ans = ans
        except OverflowError:
            final_ans = ans
        return final_ans

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Division(Variable(var), self.child).simplify()
        elif isinstance(self.child, Variable):
            return Logarithm(Absolute(self.child)).simplify()
        else:
            raise NotImplementedError('Integral too complex')

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('frac',
                                     mathml_tag('row',
                                                mathml_tag('n', '1'))
                                     + self.child.mathml()))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simple_child = self.child.simplify()
        try:
            return Nodeify(1 / simple_child.evaluate())
        except KeyError:
            pass
        if isinstance(simple_child, Invert):
            return simple_child.child
        else:
            return Invert(simple_child)

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'Divide[1, {self.child.wolfram()}]'


class Floor(UnaryOperator):
    """floor operator"""
    __slots__ = ()
    symbol = 'floor'
    wolfram_func = 'Floor'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return floor(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        raise NotImplementedError('Integration not supported for this expression')

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '⌊')
                          + self.child.mathml()
                          + mathml_tag('o', '⌋'))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simple_child = self.child.simplify()
        try:
            return Nodeify(floor(simple_child.evaluate()))
        except KeyError:
            pass
        if isinstance(simple_child, (Floor, Ceiling)):
            return simple_child
        else:
            return Floor(simple_child)

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}]'


class Ceiling(UnaryOperator):
    """ceiling operator"""
    __slots__ = ()
    symbol = 'ceil'
    wolfram_func = 'Ceiling'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Integer(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return ceil(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        raise NotImplementedError('Integration not supported for this expression')

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '⌈')
                          + self.child.mathml()
                          + mathml_tag('o', '⌉'))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simple_child = self.child.simplify()
        try:
            return Nodeify(ceil(simple_child.evaluate()))
        except KeyError:
            pass
        if isinstance(simple_child, (Floor, Ceiling)):
            return simple_child
        else:
            return Ceiling(simple_child)

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}]'


class Not(UnaryOperator):
    """Logical not operator"""
    __slots__ = ()
    symbol = '~'
    wolfram_func = 'Not'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Not(self.child.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> bool:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return not self.child.evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Piecewise([(Variable(var), self)]).simplify()

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

    # todo: reimplement Not.simplify
    # def simplify(self) -> 'Node':
    #     """returns a simplified version of the tree"""
    #     simple_child = self.child.simplify()
    #     try:
    #         return Constant(simple_child.evaluate())
    #     except KeyError:
    #         pass
    #     if isinstance(simple_child, Not):
    #         return simple_child.child
    #     elif isinstance(simple_child, Nand):
    #         return And(simple_child.child1, simple_child.child2).simplify()
    #     elif isinstance(simple_child, Nor):
    #         return Or(simple_child.child1, simple_child.child2).simplify()
    #     elif isinstance(simple_child, Xnor):
    #         return Xor(simple_child.child1, simple_child.child2).simplify()
    #     elif isinstance(simple_child, And):
    #         return Nand(simple_child.child1, simple_child.child2).simplify()
    #     elif isinstance(simple_child, Or):
    #         return Nor(simple_child.child1, simple_child.child2).simplify()
    #     elif isinstance(simple_child, Xor):
    #         return Xnor(simple_child.child1, simple_child.child2).simplify()
    #     else:
    #         return Not(simple_child)


class CalculusOperator(Node, metaclass=ABCMeta):
    """Calculus-related operator nodes"""
    __slots__ = ('child', 'variable')
    wolfram_func = ''
    symbol = ''

    def __init__(self, expression: 'Node', variable: 'str') -> None:
        self.child = expression.copy()
        self.variable = variable
        super().__init__()

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.child, self.variable)

    def list_nodes(self) -> list['Node']:
        """returns a list of all nodes in the tree"""
        out = [self]  # type: list[Node]
        return out + self.child.list_nodes()

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        return self.__class__(self.child.simplify(), self.variable)

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child.substitute(var, sub), self.variable)


class Derivative(CalculusOperator):
    """Derivative operation node"""
    __slots__ = ()
    wolfram_func = 'D'

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return self.child.derivative(self.variable).dependencies()

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Derivative(self, variable)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child.derivative(self.variable).evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'd({self.child})/d{self.variable}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var == self.variable:
            return self.child.copy().simplify()
        else:
            return IndefiniteIntegral(self, var).simplify()

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

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        result = self.child.simplify().derivative(self.variable)
        if isinstance(result, (IndefiniteIntegral, DefiniteIntegral)):
            return result
        else:
            return result.simplify()

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}, {self.variable}]'


class IndefiniteIntegral(CalculusOperator):
    """Indefinite Integral operator node"""
    wolfram_func = 'Integrate'
    __slots__ = ()

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies() | {self.variable, }

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if variable == self.variable:
            return self.child.copy()
        else:
            return Derivative(self, variable)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child.simplify().integral(self.variable).evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'[{self.child.infix()}]d{self.variable}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return IndefiniteIntegral(self, var).simplify()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('o', '&int;')
                          + self.child.mathml()
                          + mathml_tag('i', 'd')
                          + mathml_tag('i', self.variable))

    # todo: reimplement IndefiniteIntegral.simplify
    # def simplify(self) -> 'Node':
    #     """returns a simplified version of the tree"""
    #     simple_child = self.child.simplify()
    #     try:
    #         return simple_child.integral(self.variable).simplify()
    #     except NotImplementedError:
    #         if isinstance(simple_child, Sum):
    #             return Sum(self.__class__(simple_child.child1, self.variable),
    #                        self.__class__(simple_child.child2, self.variable))
    #         elif isinstance(simple_child, Subtraction):
    #             return Subtraction(self.__class__(simple_child.child1, self.variable),
    #                                self.__class__(simple_child.child2, self.variable))
    #         else:
    #             return self.__class__(self.child.simplify(), self.variable)

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}, {self.variable}]'


class DefiniteIntegral(CalculusOperator):
    """Definite Integral operator node"""
    wolfram_func = 'Integrate'
    symbol = 'dint'
    __slots__ = ('lower', 'upper')

    def __init__(self, expression: 'Node', variable: str, lower: 'Node', upper: 'Node') -> None:
        super().__init__(expression, variable)
        self.lower = lower.copy()
        self.upper = upper.copy()

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.child, self.variable, self.lower, self.upper)

    def dependencies(self) -> set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies() - {self.variable, }

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if variable == self.variable:
            return Integer(0)
        else:
            return Derivative(self, variable)

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        indefinite = self.child.integral(self.variable)
        return (indefinite.substitute(self.variable, self.upper).evaluate(var_dict)
                - indefinite.substitute(self.variable, self.lower).evaluate(var_dict))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'[∫_{self.lower.infix()}^{self.upper.infix()} {self.child.infix()}]d{self.variable}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return IndefiniteIntegral(self, var).simplify()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mathml_tag('row',
                          mathml_tag('subsup',
                                     mathml_tag('o', '&int;')
                                     + self.lower.mathml()
                                     + self.upper.mathml())
                          + self.child.mathml()
                          + mathml_tag('i', 'd')
                          + mathml_tag('i', self.variable))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        return self.__class__(self.child.simplify(), self.symbol, self.lower.simplify(), self.upper.simplify())

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()},' \
               f'{{{self.variable}, {self.lower.wolfram()}, {self.upper.wolfram()}}}]'


class Piecewise(Node):
    """Piecewise function node"""
    wolfram_func = 'Piecewise'
    symbol = 'piecewise'
    __slots__ = 'expressions', 'default'

    def __init__(self, expressions: list[tuple[Node, Node]], default: Optional[Node] = None):
        self.default = default.copy() if default is not None else Integer(0)
        self.expressions = [(expr.copy(), cond.copy()) for expr, cond in expressions]
        super().__init__()

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return Piecewise([(expr, cond) for expr, cond in self.expressions], self.default)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Piecewise([(expr.derivative(variable), cond) for expr, cond in self.expressions],
                         self.default.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> ConstantType:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        for expression, condition in self.expressions:
            if condition.evaluate(var_dict):
                return expression.evaluate(var_dict)
        else:
            return self.default.evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        expression_part = ''
        for expr, cond in self.expressions:
            expression_part += f'({expr.infix()}, {cond.infix()}), '
        expression_part += self.default.infix()
        return self.symbol + '(' + expression_part + ')'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Piecewise([(expr.integral(var), cond) for expr, cond in self.expressions],
                         self.default.integral(var)).simplify()

    def list_nodes(self) -> list['Node']:
        """returns a list of all nodes in the tree"""
        out = [self]  # type: list[Node]
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

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        return self.__class__([(x.simplify(), y.simplify()) for x, y in self.expressions],
                              self.default.simplify())

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return Piecewise([(expr.substitute(var, sub), cond.substitute(var, sub)) for expr, cond in self.expressions],
                         self.default.substitute(var, sub))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        expressions = ', '.join(f'{{{expr.wolfram()}, {cond.wolfram()}}}' for expr, cond in self.expressions)
        return f'{self.wolfram_func}[{{{expressions}}}, {self.default.wolfram()}]'

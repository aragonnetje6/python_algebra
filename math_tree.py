"""
basic expression tree with evaluation and derivation
"""

from abc import ABCMeta, abstractmethod
from itertools import combinations_with_replacement
from math import e, log, sin, cos, tan, asin, acos, atan, isclose
from typing import Optional, Dict, Union, Tuple, List, Set

Number = Union[int, float]
Variables = Dict[str, Number]
Expression = Union[Tuple[Union[tuple, Number, str], Union[tuple, Number, str], Union[tuple, Number, str], Union[
    tuple, Number, str], str],
                   Tuple[Union[tuple, Number, str], Union[tuple, Number, str], str],
                   Tuple[Union[tuple, Number, str], str],
                   Number,
                   str]


def tag(xml_tag: str, content: str, args: Optional[str] = None):
    """XML tag wrapping function"""
    if args is None:
        return f'<{xml_tag}>{content}</{xml_tag}>'
    else:
        return f'<{xml_tag} {args}>{content}</{xml_tag}>'


def mtag(xml_tag: str, content: str, args: Optional[str] = None):
    """Mathml tag wrapping function"""
    return tag('m' + xml_tag, content, args)


class Node(metaclass=ABCMeta):
    """Abstract Base Class for any node in the expression tree"""
    __slots__ = 'parent',

    def __init__(self) -> None:
        self.parent: Optional[Node] = None
        self.reset_parents()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __str__(self) -> str:
        return self.infix()

    def __eq__(self, other: 'Node') -> 'Equal':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Equal(self, other)
        else:
            return NotImplemented

    def __gt__(self, other: 'Node') -> 'GreaterThan':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return GreaterThan(self, other)
        else:
            return NotImplemented

    def __ge__(self, other: 'Node') -> 'GreaterEqual':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return GreaterEqual(self, other)
        else:
            return NotImplemented

    def __lt__(self, other: 'Node') -> 'LessThan':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return LessThan(self, other)
        else:
            return NotImplemented

    def __le__(self, other: 'Node') -> 'LessEqual':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return LessEqual(self, other)
        else:
            return NotImplemented

    def __add__(self, other: Union[str, Number, 'Node']) -> 'Addition':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Addition(self, other)
        else:
            return NotImplemented

    def __radd__(self, other: Union[str, Number, 'Node']) -> 'Addition':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Addition(other, self)
        else:
            return NotImplemented

    def __sub__(self, other: Union[str, Number, 'Node']) -> 'Subtraction':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Subtraction(self, other)
        else:
            return NotImplemented

    def __rsub__(self, other: Union[str, Number, 'Node']) -> 'Subtraction':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Subtraction(other, self)
        else:
            return NotImplemented

    def __mul__(self, other: Union[str, Number, 'Node']) -> 'Product':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Product(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other: Union[str, Number, 'Node']) -> 'Product':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Product(other, self)
        else:
            return NotImplemented

    def __truediv__(self, other: Union[str, Number, 'Node']) -> 'Division':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Division(self, other)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Union[str, Number, 'Node']) -> 'Division':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Division(other, self)
        else:
            return NotImplemented

    def __pow__(self, other: Union[str, Number, 'Node']) -> 'Exponent':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Exponent(self, other)
        else:
            return NotImplemented

    def __rpow__(self, other: Union[str, Number, 'Node']) -> 'Exponent':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Exponent(other, self)
        else:
            return NotImplemented

    def __neg__(self) -> 'Negate':
        return Negate(self)

    def __invert__(self) -> 'Not':
        return Not(self)

    def __and__(self, other: Union[str, Number, 'Node']) -> 'And':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return And(self, other)
        else:
            return NotImplemented

    def __rand__(self, other: Union[str, Number, 'Node']) -> 'And':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return And(other, self)
        else:
            return NotImplemented

    def __or__(self, other: Union[str, Number, 'Node']) -> 'Or':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Or(self, other)
        else:
            return NotImplemented

    def __ror__(self, other: Union[str, Number, 'Node']) -> 'Or':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Or(other, self)
        else:
            return NotImplemented

    def __xor__(self, other: Union[str, Number, 'Node']) -> 'Xor':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Xor(self, other)
        else:
            return NotImplemented

    def __rxor__(self, other: Union[str, Number, 'Node']) -> 'Xor':
        if isinstance(other, str):
            other = Variable(other)
        elif isinstance(other, (float, int)):
            other = Constant(other)
        if isinstance(other, Node):
            return Xor(other, self)
        else:
            return NotImplemented

    @staticmethod
    def dependencies() -> Set[str]:
        """returns set of all variables present in the tree"""
        return set()

    @abstractmethod
    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""

    @abstractmethod
    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""

    @abstractmethod
    def infix(self) -> str:
        """returns infix representation of the tree"""

    @abstractmethod
    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""

    @abstractmethod
    def latex(self) -> str:
        """return latex language representation of the tree"""

    @abstractmethod
    def list_nodes(self) -> List['Node']:
        """return latex language representation of the tree"""

    @abstractmethod
    def mathml(self) -> str:
        """returns the MathML representation of the tree"""

    @abstractmethod
    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""

    @abstractmethod
    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""

    @abstractmethod
    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""

    @abstractmethod
    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""

    @abstractmethod
    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__()

    def get_root(self) -> 'Node':
        """Returns the root node of the expression tree"""
        if self.parent is None:
            return self
        return self.parent.get_root()

    def plot(self, var: str, minimum: Number, maximum: Number, var_dict: Optional[Variables] = None, n: int = 10000,
             *args, **kwargs):
        """Plot function over supplied variable range"""
        import matplotlib.pyplot as plt

        if var_dict is None:
            var_dict = {}

        x = [i / n * (maximum - minimum) + minimum for i in range(n)]
        y = []
        for i in x:
            try:
                y.append(self.evaluate({**var_dict, var: i}))
            except (ArithmeticError, NotImplementedError):
                y.append(None)

        plt.plot(x, y, *args, **kwargs)
        plt.show()

    def reset_parents(self, parent: Optional['Node'] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        self.parent = parent

    def total_derivative(self) -> 'Node':
        """
        returns an expression tree representing the total derivative of this tree.
        the total derivative of f is defined as sum(f.derivative(var) for var in f.dependencies)
        """
        out: Node = Constant(0)
        for variable in self.dependencies():
            out = Addition(out, self.derivative(variable))
        out.reset_parents()
        return out


class Term(Node, metaclass=ABCMeta):
    """Abstract Base Class for any value (leaf node) in the expression tree"""
    __slots__ = 'value',

    def __init__(self, value: Union[str, Number]) -> None:
        super().__init__()
        self.value = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.value)

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return str(self.value)

    def list_nodes(self) -> List[Node]:
        """returns a list of all nodes in the tree"""
        return [self]

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return str(self.value)

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        return self.copy()

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.value


class Constant(Term):
    """Real numerical constant in expression tree, """
    __slots__ = ()

    def __init__(self, value: Number) -> None:
        assert isinstance(value, (int, float))
        super().__init__(value)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Constant(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var))

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('n',
                         str(self.value)))

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.copy()

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'Real[{self.value}]'


class Variable(Term):
    """Named variable in expression tree"""
    __slots__ = ()

    def __init__(self, value: str) -> None:
        assert isinstance(value, str)
        super().__init__(value)

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return {self.value}

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if self.value == variable:
            return Constant(1)
        return Constant(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        if var_dict is None:
            raise KeyError(f'None does not contain "{self.value}"')
        return var_dict[self.value]

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if self.value == var:
            if self.parent is None or isinstance(self.parent, (Addition, Subtraction)):
                return Division(Exponent(self, Constant(2)), Constant(2))
            else:
                return self.copy()
        else:
            return Constant(0)

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('i',
                         str(self.value)))

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        if self.value == var:
            return sub.copy()
        return self.copy()

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return self.value


class BinaryOperator(Node, metaclass=ABCMeta):
    """Abstract Base Class for 2-input operator in expression tree"""
    __slots__ = 'child1', 'child2',
    symbol = ''
    wolfram_func = ''

    def __init__(self, child1: Node, child2: Node) -> None:
        assert isinstance(child1, Node)
        assert isinstance(child2, Node)
        self.child1 = child1.copy()
        self.child2 = child2.copy()
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child1)}, {repr(self.child2)})'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.child1, self.child2)

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return self.child1.dependencies().union(self.child2.dependencies())

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if (isinstance(self.parent, Division) and self.parent.child2 is self) or isinstance(self.parent, Exponent):
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'

    def latex(self) -> str:
        """return latex language representation of the tree"""
        if (isinstance(self.parent, Division) and self.parent.child2 is self) or isinstance(self.parent, Exponent):
            return f'({self.child1.latex()} {self.symbol} {self.child2.latex()})'
        else:
            return f'{self.child1.latex()} {self.symbol} {self.child2.latex()}'

    def list_nodes(self) -> List[Node]:
        """returns a list of all nodes in the tree"""
        out = [self]  # type: List[Node]
        return out + self.child1.list_nodes() + self.child2.list_nodes()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if (isinstance(self.parent, Division) and self.parent.child2 is self) or isinstance(self.parent, Exponent):
            return mtag('row',
                        mtag('fenced',
                             mtag('row',
                                  self.child1.mathml()
                                  + mtag('o', self.symbol)
                                  + self.child2.mathml())))
        else:
            return mtag('row',
                        self.child1.mathml()
                        + mtag('o', self.symbol)
                        + self.child2.mathml())

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        self.child1.reset_parents(self)
        self.child2.reset_parents(self)

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.child1.rpn() + ' ' + self.child2.rpn() + ' ' + self.symbol

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        else:
            return self.__class__(self.child1.simplify(), self.child2.simplify())

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child1.substitute(var, sub), self.child2.substitute(var, sub))

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child1.tuple(), self.child2.tuple(), self.symbol

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child1.wolfram()}, {self.child2.wolfram()}]'


class Addition(BinaryOperator):
    """Addition operator node"""
    __slots__ = ()
    symbol = '+'
    wolfram_func = 'Plus'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Addition(self.child1.derivative(variable), self.child2.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child1.evaluate(var_dict) + self.child2.evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, UnaryOperator)) or (
                isinstance(self.parent, Subtraction) and self.parent.child1 is self):
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'
        else:
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var in self.dependencies():
            return Addition(self.child1.integral(var),
                            self.child2.integral(var))

    def latex(self) -> str:
        """return latex language representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, UnaryOperator)) or (
                isinstance(self.parent, Subtraction) and self.parent.child1 is self):
            return f'{self.child1.latex()} {self.symbol} {self.child2.latex()}'
        else:
            return f'({self.child1.latex()} {self.symbol} {self.child2.latex()})'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, UnaryOperator)) or (
                isinstance(self.parent, Subtraction) and self.parent.child1 is self):
            return mtag('row',
                        self.child1.mathml()
                        + mtag('o', self.symbol)
                        + self.child2.mathml())
        else:
            return mtag('row',
                        mtag('fenced',
                             mtag('row',
                                  self.child1.mathml()
                                  + mtag('o', self.symbol)
                                  + self.child2.mathml())))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify()
        child2 = self.child2.simplify()
        if isinstance(child1, Constant) and isinstance(child2, Constant):
            return Constant(child1.evaluate() + child2.evaluate())
        elif isinstance(child1, Constant) and child1.evaluate() == 0:
            return child2
        elif isinstance(child2, Constant) and child2.evaluate() == 0:
            return child1
        else:
            return Addition(child1, child2)


class Subtraction(BinaryOperator):
    """Subtraction operator node"""
    __slots__ = ()
    symbol = '-'
    wolfram_func = 'Subtract'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Subtraction(self.child1.derivative(variable), self.child2.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child1.evaluate(var_dict) - self.child2.evaluate(var_dict)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Subtraction(self.child1.integral(var),
                           self.child2.integral(var))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify()
        child2 = self.child2.simplify()
        if isinstance(child1, Constant) and isinstance(child2, Constant):
            return Constant(child1.evaluate() - child2.evaluate())
        elif isinstance(child1, Constant) and child1.evaluate() == 0:
            return Negate(child2)
        elif isinstance(child2, Constant) and child2.evaluate() == 0:
            return child1
        return Subtraction(child1, child2)


class Product(BinaryOperator):
    """Multiplication operator node"""
    __slots__ = ()
    symbol = '*'
    wolfram_func = 'Times'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Addition(Product(self.child1, self.child2.derivative(variable)),
                        Product(self.child1.derivative(variable), self.child2))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        try:
            return self.child1.evaluate(var_dict) * self.child2.evaluate(var_dict)
        except (ArithmeticError, ValueError) as err:
            try:
                ans1 = self.child1.evaluate(var_dict)
                if ans1 == 0:
                    return 0
                else:
                    raise err from err
            except (ArithmeticError, ValueError) as err:
                ans2 = self.child2.evaluate(var_dict)
                if ans2 == 0:
                    return 0
                else:
                    raise err from err

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif var not in self.child1.dependencies():
            return Product(self.child1, self.child2.integral(var))
        elif var not in self.child2.dependencies():
            return Product(self.child2, self.child1.integral(var))
        else:
            raise NotImplementedError('Integration not supported for this expression')

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify()
        child2 = self.child2.simplify()
        if isinstance(child1, Constant) and isinstance(child2, Constant):
            return Constant(child1.evaluate() * child2.evaluate())
        elif isinstance(child1, Constant):
            if child1.evaluate() == 0:
                return Constant(0)
            elif child1.evaluate() == 1:
                return child2
        elif isinstance(child2, Constant):
            if child2.evaluate() == 0:
                return Constant(0)
            elif child2.evaluate() == 1:
                return child1
        return Product(child1, child2)


class Division(BinaryOperator):
    """Division operator node"""
    __slots__ = ()
    symbol = '/'
    wolfram_func = 'Divide'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(Subtraction(Product(self.child1.derivative(variable), self.child2),
                                    Product(self.child1, self.child2.derivative(variable))),
                        Exponent(self.child2, Constant(2)))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child1.evaluate(var_dict) / self.child2.evaluate(var_dict)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif var not in self.child1.dependencies() and self.child2 == Variable(var):
            return Product(Logarithm(Absolute(Variable(var)), Constant(e)), Constant(self.child1.evaluate()))
        elif var not in self.child2.dependencies():
            return Division(self.child1.integral(var), Constant(self.child1.evaluate()))
        else:
            raise NotImplementedError('Integration not supported for this expression')

    def latex(self) -> str:
        """return the latex language representation of the expression"""
        return f'\\frac{{{self.child1.latex()}}}{{{self.child2.latex()}}}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('frac',
                         self.child1.mathml()
                         + self.child2.mathml()))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify()
        child2 = self.child2.simplify()
        if isinstance(child1, Constant) and isinstance(child2, Constant):
            return Constant(child1.evaluate() / child2.evaluate())
        elif isinstance(child1, Constant):
            if child1.evaluate() == 0:
                return Constant(0)
            elif child1.evaluate() == 1:
                return Invert(child2)
            elif child1.evaluate() == -1:
                return Negate(child2)
        elif isinstance(child2, Constant):
            if child2.evaluate() == 1:
                return child1
            elif child2.evaluate() == -1:
                return Negate(child1)
            else:
                return Division(child1, child2)
        return Division(child1, child2)


class Exponent(BinaryOperator):
    """Exponent operator node"""
    __slots__ = ()
    symbol = '**'
    wolfram_func = 'Power'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Product(self,
                       Addition(Product(self.child1.derivative(variable),
                                        Division(self.child2,
                                                 self.child1)),
                                Product(self.child2.derivative(variable),
                                        Logarithm(self.child1,
                                                  Constant(e)))))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        try:
            ans: Union[Number, complex] = self.child1.evaluate(var_dict) ** self.child2.evaluate(var_dict)
            if isinstance(ans, complex):
                raise ArithmeticError('Complex values not allowed')
            else:
                return ans
        except (ArithmeticError, ValueError) as err:
            try:
                ans1 = self.child1.evaluate(var_dict)
                if ans1 == 0 or ans1 == 1:
                    return ans1
                else:
                    raise err from err
            except (ArithmeticError, ValueError) as err:
                ans2 = self.child2.evaluate(var_dict)
                if ans2 == 0:
                    return 1
                else:
                    raise err from err

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if isinstance(self.parent, Exponent):
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif var not in self.child1.dependencies() and self.child2 == Variable(var):
            return Division(Exponent(self.child1,
                                     Variable(var)),
                            Logarithm(self.child1,
                                      Constant(e)))
        elif var not in self.child2.dependencies() and self.child1 == Variable(var):
            return Division(Exponent(Variable(var),
                                     Addition(self.child2,
                                              Constant(1))),
                            Addition(self.child2,
                                     Constant(1)))
        else:
            raise NotImplementedError('Integration not supported for this expression')

    def latex(self) -> str:
        """returns latex representation of the tree"""
        if isinstance(self.parent, Exponent):
            return f'({self.child1.latex()} {self.symbol} {self.child2.latex()})'
        else:
            return f'{self.child1.latex()} {self.symbol} {self.child2.latex()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if isinstance(self.parent, Exponent):
            return mtag('row',
                        mtag('fenced',
                             mtag('row',
                                  mtag('sup',
                                       self.child1.mathml()
                                       + self.child2.mathml()))))
        else:
            return mtag('row',
                        mtag('sup',
                             self.child1.mathml()
                             + self.child2.mathml()))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify()
        child2 = self.child2.simplify()
        if isinstance(child1, Constant) and isinstance(child2, Constant):
            return Constant(child1.evaluate() ** child2.evaluate())
        elif isinstance(child1, Constant):
            if child1.evaluate() == 0:
                return Constant(0)
            elif child1.evaluate() == 1:
                return child2
        elif isinstance(child2, Constant):
            if child2.evaluate() == 0:
                return Constant(1)
            elif child2.evaluate() == 1:
                return child1
        return Exponent(child1, child2)


class Logarithm(BinaryOperator):
    """Logarithm operator node, child 2 is base"""
    __slots__ = ()
    symbol = 'log'
    wolfram_func = 'Log'

    def __init__(self, child1: Node, child2: Optional[Node] = None):
        if child2 is None:
            child2 = Constant(e)
        super().__init__(child1, child2)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return log(self.child1.evaluate(var_dict), self.child2.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(Subtraction(Division(Product(self.child1.derivative(variable),
                                                     Logarithm(self.child2,
                                                               Constant(e))),
                                             self.child1),
                                    Division(Product(self.child2.derivative(variable),
                                                     Logarithm(self.child1,
                                                               Constant(e))),
                                             self.child2)),
                        Exponent(Logarithm(self.child2,
                                           Constant(e)),
                                 Constant(2)))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'{self.symbol}({self.child1.infix()}, {self.child2.infix()})'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child1 == Variable(var) and len(self.child2.dependencies()) == 0 and self.child2.evaluate() == e:
            return Subtraction(Product(Variable(var),
                                       Logarithm(Variable(var),
                                                 Constant(e))),
                               Variable(var))
        else:
            raise NotImplementedError('Integration not supported for this expression')

    def latex(self) -> str:
        """returns latex language representation of the tree"""
        return f'\\log{{{self.child2.latex()}}}({self.child1.latex()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('sub',
                         mtag('i', self.symbol)
                         + self.child2.mathml())
                    + mtag('fenced', self.child1.mathml()))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        child1 = self.child1.simplify()
        child2 = self.child2.simplify()
        if isinstance(child1, Constant) and isinstance(child2, Constant):
            return Constant(log(child1.evaluate(), child2.evaluate()))
        elif isinstance(child1, Constant):
            if child1.evaluate() == 1:
                return Constant(0)
        return Logarithm(child1, child2)


class ComparisonLogicalOperator(BinaryOperator, metaclass=ABCMeta):
    """Abstract base class for comparison operators"""
    __slots__ = ()

    @staticmethod
    @abstractmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Constant(0)  # todo: piecewise

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        # todo: un-brute force this
        if var_dict is None:
            return int(self.comparison_function(self.child1.evaluate(), self.child2.evaluate()))
        else:
            simp = self
            for var, val in var_dict.items():
                simp = simp.substitute(var, Constant(val))
            dependencies = simp.dependencies()
            if len(dependencies) > 0:
                return int(all(
                    self.comparison_function(
                        self.child1.evaluate({letter: nr for letter, nr in zip(dependencies, values)}),
                        self.child2.evaluate({letter: nr for letter, nr in zip(dependencies, values)}))
                    for values in combinations_with_replacement([-2 ** x for x in range(20, -21, -1)]
                                                                + [2 ** x for x in range(-20, 21)],
                                                                len(dependencies))))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Piecewise([(Variable(var), self)])

    def latex(self) -> str:
        """returns latex representation of the tree"""
        if self.parent is not None:
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if self.parent is None:
            return mtag('row',
                        self.child1.mathml()
                        + mtag('o', self.symbol)
                        + self.child2.mathml())
        else:
            return mtag('row',
                        mtag('fenced',
                             mtag('row',
                                  self.child1.mathml()
                                  + mtag('o', self.symbol)
                                  + self.child2.mathml())))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child1.wolfram()}][{self.child2.wolfram()}]'


class Equal(ComparisonLogicalOperator):
    """Equality operator node"""
    __slots__ = ()
    symbol = '=='
    wolfram_func = 'EqualTo'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return isclose(x, y)


class NotEqual(ComparisonLogicalOperator):
    """Inequality operator node"""
    __slots__ = ()
    symbol = '!='
    wolfram_func = 'UnequalTo'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return x != y


class GreaterThan(ComparisonLogicalOperator):
    """Greater-than operator node"""
    __slots__ = ()
    symbol = '>'
    wolfram_func = 'GreaterThan'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return x > y


class LessThan(ComparisonLogicalOperator):
    """Less-than operator node"""
    __slots__ = ()
    symbol = '<'
    wolfram_func = 'LessThan'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return x < y


class GreaterEqual(ComparisonLogicalOperator):
    """Greater-equal operator node"""
    __slots__ = ()
    symbol = '>='
    wolfram_func = 'GreaterEqual'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return x >= y


class LessEqual(ComparisonLogicalOperator):
    """Less-equal operator node"""
    __slots__ = ()
    symbol = '<='
    wolfram_func = 'LessEqual'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return x <= y


class And(ComparisonLogicalOperator):
    """logical AND operator node"""
    __slots__ = ()
    symbol = '&'
    wolfram_func = 'And'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return bool(x) & bool(y)

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        simp_child1 = self.child1.simplify()
        simp_child2 = self.child2.simplify()
        if isinstance(simp_child1, Not) and isinstance(simp_child2, Not):
            return Nor(simp_child1.child, simp_child2.child).simplify()
        else:
            return And(simp_child1, simp_child2)


class Or(ComparisonLogicalOperator):
    """logical OR operator node"""
    __slots__ = ()
    symbol = '|'
    wolfram_func = 'Or'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return bool(x) | bool(y)

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        simp_child1 = self.child1.simplify()
        simp_child2 = self.child2.simplify()
        if isinstance(simp_child1, Not) and isinstance(simp_child2, Not):
            return Nand(simp_child1.child, simp_child2.child).simplify()
        else:
            return Nor(simp_child1, simp_child2)


class Nand(ComparisonLogicalOperator):
    """logical NAND operator node"""
    __slots__ = ()
    wolfram_func = 'Nand'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return not (bool(x) & bool(y))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return f'(not (bool({self.child1.infix()}) & bool({self.child2.infix()})))'
        else:
            return f'not (bool({self.child1.infix()}) & bool({self.child2.infix()}))'

    def latex(self) -> str:
        """returns latex representation of the tree"""
        return f'~({self.child1.infix()} & {self.child2.infix()})'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('o', '~')
                    + mtag('fenced',
                           mtag('row',
                                self.child1.mathml()
                                + mtag('o', '&')
                                + self.child2.mathml())))

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        simp_child1 = self.child1.simplify()
        simp_child2 = self.child2.simplify()
        if isinstance(simp_child1, Not) and isinstance(simp_child2, Not):
            return Or(simp_child1.child, simp_child2.child).simplify()
        else:
            return Nand(simp_child1, simp_child2)


class Nor(ComparisonLogicalOperator):
    """logical NOR operator node"""
    __slots__ = ()
    wolfram_func = 'Nor'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return not (bool(x) | bool(y))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return f'(not (bool({self.child1.infix()}) | bool({self.child2.infix()})))'
        else:
            return f'not (bool({self.child1.infix()}) | bool({self.child2.infix()}))'

    def latex(self) -> str:
        """returns latex representation of the tree"""
        return f'~({self.child1.infix()} | {self.child2.infix()})'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('o', '~')
                    + mtag('fenced',
                           mtag('row',
                                self.child1.mathml()
                                + mtag('o', '|')
                                + self.child2.mathml())))

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        simp_child1 = self.child1.simplify()
        simp_child2 = self.child2.simplify()
        if isinstance(simp_child1, Not) and isinstance(simp_child2, Not):
            return And(simp_child1.child, simp_child2.child).simplify()
        else:
            return Nor(simp_child1, simp_child2)


class Xor(ComparisonLogicalOperator):
    """logical XOR operator node"""
    __slots__ = ()
    symbol = '^'
    wolfram_func = 'Xor'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return bool(x) ^ bool(y)

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        simp_child1 = self.child1.simplify()
        simp_child2 = self.child2.simplify()
        if isinstance(simp_child1, Not) and isinstance(simp_child2, Not):
            return Xor(simp_child1.child, simp_child2.child).simplify()
        else:
            return Xor(simp_child1, simp_child2)


class Xnor(ComparisonLogicalOperator):
    """logical XOR operator node"""
    __slots__ = ()
    wolfram_func = 'Xnor'

    @staticmethod
    def comparison_function(x: Number, y: Number) -> bool:
        """Compare both numbers"""
        return not (bool(x) ^ bool(y))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is not None:
            return f'(not (bool({self.child1.infix()}) ^ bool({self.child2.infix()})))'
        else:
            return f'not (bool({self.child1.infix()}) ^ bool({self.child2.infix()}))'

    def latex(self) -> str:
        """returns latex representation of the tree"""
        return f'~({self.child1.infix()} ^ {self.child2.infix()})'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('o', '~')
                    + mtag('fenced',
                           mtag('row',
                                self.child1.mathml()
                                + mtag('o', '^')
                                + self.child2.mathml())))

    def simplify(self) -> Node:
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        simp_child1 = self.child1.simplify()
        simp_child2 = self.child2.simplify()
        if isinstance(simp_child1, Not) and isinstance(simp_child2, Not):
            return Xnor(simp_child1.child, simp_child2.child).simplify()
        else:
            return Xnor(simp_child1, simp_child2)


class UnaryOperator(Node, metaclass=ABCMeta):
    """Abstract Base Class for single-input operator in expression tree"""
    __slots__ = 'child',
    symbol = ''
    wolfram_func = ''
    latex_func = ''

    def __init__(self, child: Node) -> None:
        assert isinstance(child, Node)
        self.child = child.copy()
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child)})'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.child)

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies()

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'{self.symbol}({self.child.infix()})'

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return f'{self.latex_func}({self.child.latex()})'

    def list_nodes(self) -> List[Node]:
        """returns a list of all nodes in the tree"""
        out = [self]  # type: List[Node]
        return out + self.child.list_nodes()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('i', self.symbol)
                    + mtag('fenced', self.child.mathml()))

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        self.child.reset_parents(self)

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.child.rpn() + ' ' + self.symbol

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        if len(self.dependencies()) == 0:
            return Constant(self.evaluate())
        else:
            return self.__class__(self.child.simplify())

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child.substitute(var, sub))

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child.tuple(), self.symbol

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}]'


class Sine(UnaryOperator):
    """Sine operator node in radians"""
    __slots__ = ()
    symbol = 'sin'
    wolfram_func = 'Sin'
    latex_func = '\\sin'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Product(Cosine(self.child),
                       self.child.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return sin(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child == Variable(var):
            return Product(Constant(-1),
                           Cosine(Variable(var)))
        else:
            raise NotImplementedError('Integration not supported for this expression')


class Cosine(UnaryOperator):
    """Cosine operator node in radians"""
    __slots__ = ()
    symbol = 'cos'
    wolfram_func = 'Cos'
    latex_func = '\\cos'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Subtraction(Constant(0),
                           Product(Sine(self.child),
                                   self.child.derivative(variable)))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return cos(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child == Variable(var):
            return Sine(Variable(var))
        else:
            raise NotImplementedError('Integration not supported for this expression')


class Tangent(UnaryOperator):
    """Tangent operator node in radians"""
    __slots__ = ()
    symbol = 'tan'
    wolfram_func = 'Tan'
    latex_func = '\\tan'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Cosine(self.child),
                                 Constant(2)))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return tan(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child == Variable(var):
            return Product(Constant(-1),
                           Logarithm(Cosine(Variable(var)),
                                     Constant(e)))
        else:
            raise NotImplementedError('Integration not supported for this expression')


class ArcSine(UnaryOperator):
    """Arcsine operator node in radians"""
    __slots__ = ()
    symbol = 'asin'
    wolfram_func = 'ArcSin'
    latex_func = '\\arcsin'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Subtraction(Constant(1),
                                             Exponent(self.child,
                                                      Constant(2))),
                                 Constant(1 / 2)))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return asin(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child == Variable(var):
            return Addition(Product(Variable(var),
                                    self),
                            Exponent(Subtraction(Constant(1),
                                                 Exponent(Variable(var),
                                                          Constant(2))),
                                     Constant(1 / 2)))
        else:
            raise NotImplementedError('Integration not supported for this expression')


class ArcCosine(UnaryOperator):
    """Arccosine operator node in radians"""
    __slots__ = ()
    symbol = 'acos'
    wolfram_func = 'ArcCos'
    latex_func = '\\arccos'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Subtraction(Constant(0),
                           Division(self.child.derivative(variable),
                                    Exponent(Subtraction(Constant(1),
                                                         Exponent(self.child,
                                                                  Constant(2))),
                                             Constant(1 / 2))))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return acos(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child == Variable(var):
            return Subtraction(Product(Variable(var),
                                       self),
                               Exponent(Subtraction(Constant(1),
                                                    Exponent(Variable(var),
                                                             Constant(2))),
                                        Constant(1 / 2)))
        else:
            raise NotImplementedError('Integration not supported for this expression')


class ArcTangent(UnaryOperator):
    """Arctangent operator node in radians"""
    __slots__ = ()
    symbol = 'atan'
    wolfram_func = 'ArcTan'
    latex_func = '\\arctan'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Addition(Constant(1),
                                 Exponent(self.child,
                                          Constant(2))))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return atan(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child == Variable(var):
            return Addition(Product(Variable(var),
                                    self),
                            Product(Constant(1 / 2),
                                    Logarithm(Addition(Exponent(Variable(var),
                                                                Constant(2)),
                                                       Constant(1)),
                                              Constant(e))))
        else:
            raise NotImplementedError('Integration not supported for this expression')


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

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return abs(self.child.evaluate(var_dict))

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Product(self, Variable(var))
        elif self.child == Variable(var):
            return Division(Product(Variable(var),
                                    Absolute(Variable(var))),
                            Constant(2))
        else:
            raise NotImplementedError('Integration not supported for this expression')

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return f'|{self.child}|'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('o', '|')
                    + self.child.mathml()
                    + mtag('o', '|'))


class Negate(UnaryOperator):
    """Unary negative operator"""
    __slots__ = ()
    symbol = '-'
    wolfram_func = 'Minus'
    latex_func = '-'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Negate(self.child.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
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
        return Negate(self.child.integral(var))

    def latex(self) -> str:
        """return latex language representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.latex_func}({self.child.latex()})'
        else:
            return f'{self.latex_func}{self.child.latex()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return mtag('row',
                        mtag('i', self.symbol)
                        + mtag('fenced', self.child.mathml()))
        else:
            return mtag('row',
                        mtag('i', self.symbol)
                        + self.child.mathml())

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simpchild = self.child.simplify()
        if isinstance(simpchild, Constant):
            return Constant(-simpchild.evaluate())
        elif isinstance(simpchild, Negate):
            return simpchild.child
        else:
            return Negate(simpchild)


class Invert(UnaryOperator):
    """Unary inversion operator"""
    __slots__ = ()
    symbol = '1/'
    latex_func = '\\frac'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if variable in self.dependencies():
            return Division(self.child.derivative(variable), Exponent(self.child, Constant(2)))
        else:
            return Constant(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return 1 / self.child.evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var not in self.dependencies():
            return Division(Variable(var), self.child)
        elif isinstance(self.child, Variable):
            return Logarithm(Absolute(self.child))
        else:
            raise NotImplementedError('Integral too complex')

    def latex(self) -> str:
        """return the latex language representation of the expression"""
        return f'\\frac{{1}}{{{self.child.latex()}}}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('frac',
                         tag('row',
                             tag('n', '1'))
                         + self.child.mathml()))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simpchild = self.child.simplify()
        if isinstance(simpchild, Constant):
            return Constant(1 / simpchild.evaluate())
        elif isinstance(simpchild, Invert):
            return simpchild.child
        elif isinstance(simpchild, Division):
            return Division(simpchild.child2, simpchild.child1)
        else:
            return Invert(simpchild)

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'Divide[1, {self.child.wolfram()}]'


class Not(UnaryOperator):
    """Logical not operator"""
    __slots__ = ()
    symbol = '~'
    wolfram_func = 'Not'
    latex_func = '~'

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Not(self.child.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return int(not self.child.evaluate(var_dict))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.symbol}({self.child.infix()})'
        else:
            return f'{self.symbol}{self.child.infix()}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Piecewise([(Variable(var), self)])

    def latex(self) -> str:
        """return latex language representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return f'{self.latex_func}({self.child.latex()})'
        else:
            return f'{self.latex_func}{self.child.latex()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if len(self.child.list_nodes()) > 1:
            return mtag('row',
                        mtag('i', self.symbol)
                        + mtag('fenced', self.child.mathml()))
        else:
            return mtag('row',
                        mtag('i', self.symbol)
                        + self.child.mathml())

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simpchild = self.child.simplify()
        if isinstance(simpchild, Not):
            return simpchild
        elif isinstance(simpchild, Nand):
            return And(simpchild.child1, simpchild.child2).simplify()
        elif isinstance(simpchild, Nor):
            return Or(simpchild.child1, simpchild.child2).simplify()
        elif isinstance(simpchild, Xnor):
            return Xor(simpchild.child1, simpchild.child2).simplify()
        elif isinstance(simpchild, And):
            return Nand(simpchild.child1, simpchild.child2).simplify()
        elif isinstance(simpchild, Or):
            return Nor(simpchild.child1, simpchild.child2).simplify()
        elif isinstance(simpchild, Xor):
            return Xnor(simpchild.child1, simpchild.child2).simplify()
        else:
            return Not(simpchild)


class CalculusOperator(Node, metaclass=ABCMeta):
    """Calculus-related operator nodes"""
    __slots__ = ('child', 'variable')
    wolfram_func = ''
    symbol = ''

    def __init__(self, expression: 'Node', variable: 'str') -> None:
        super().__init__()
        self.child = expression.copy()
        self.variable = variable

    def list_nodes(self) -> List['Node']:
        """return latex language representation of the tree"""
        out = [self]  # type: List[Node]
        return out + self.child.list_nodes()

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.child.rpn() + ' ' + self.variable + ' ' + self.symbol

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child.substitute(var, sub), self.variable)

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child.tuple(), self.variable, self.symbol

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}, {self.variable}]'


class Derivative(CalculusOperator):
    """Derivative operation node"""
    __slots__ = ()
    wolfram_func = 'Derivative'
    symbol = 'deriv'

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return self.child.derivative(self.variable).dependencies()

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Derivative(self, variable)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child.derivative(self.variable).evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'd({self.child})/d{self.variable}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if var == self.variable:
            return self.child.copy()
        else:
            return IndefiniteIntegral(self, var)

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return f'\\frac{{d}}{{d{self.variable}}}\\left({self.child.latex()}\\right)'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('frac',
                         mtag('row',
                              mtag('i',
                                   'd'))
                         + mtag('row',
                                mtag('i', 'd')
                                + mtag('i', self.variable)))
                    + mtag('fenced', self.child.mathml()))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        result = self.child.simplify().derivative(self.variable)
        if isinstance(result, (IndefiniteIntegral, DefiniteIntegral)):
            return result
        else:
            return result.simplify()


class IndefiniteIntegral(CalculusOperator):
    """Indefinite Integral operator node"""
    wolfram_func = 'Integrate'
    symbol = 'iint'
    __slots__ = ()

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies() | {self.variable, }

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if variable == self.variable:
            return self.child.copy()
        else:
            return Derivative(self, variable)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child.simplify().integral(self.variable).evaluate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'[{self.child.infix()}]d{self.variable}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return IndefiniteIntegral(self, var)

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return f'\\int\\left[{self.child.latex()}\\right]d{self.variable}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('o', '&int;')
                    + self.child.mathml()
                    + mtag('i', 'd')
                    + mtag('i', self.variable))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        simp_child = self.child.simplify()
        try:
            return simp_child.integral(self.variable).simplify()
        except NotImplementedError:
            if isinstance(simp_child, Addition):
                return Addition(self.__class__(simp_child.child1, self.variable),
                                self.__class__(simp_child.child2, self.variable))
            elif isinstance(simp_child, Subtraction):
                return Subtraction(self.__class__(simp_child.child1, self.variable),
                                   self.__class__(simp_child.child2, self.variable))
            else:
                return self.__class__(self.child.simplify(), self.variable)


class DefiniteIntegral(CalculusOperator):
    """Definite Integral operator node"""
    wolfram_func = 'Integrate'
    symbol = 'dint'
    __slots__ = ('lower', 'upper')

    def __init__(self, expression: 'Node', variable: str, lower: 'Node', upper: 'Node') -> None:
        super().__init__(expression, variable)
        self.lower = lower.copy()
        self.upper = upper.copy()

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies() - {self.variable, }

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if variable == self.variable:
            return Constant(0)
        else:
            return Derivative(self, variable)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        indefinite = self.child.integral(self.variable)
        return (indefinite.substitute(self.variable, self.upper).evaluate(var_dict)
                - indefinite.substitute(self.variable, self.lower).evaluate(var_dict))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'[∫_{self.lower.infix()}^{self.upper.infix()} {self.child.infix()}]d{self.variable}'

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return IndefiniteIntegral(self, var)

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return f'\\int_{{{self.lower.latex()}}}^{{{self.upper.latex()}}}' \
               f'\\left[{self.child.latex()}\\right]d{self.variable}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return mtag('row',
                    mtag('subsup',
                         mtag('o', '&int;')
                         + self.lower.mathml()
                         + self.upper.mathml())
                    + self.child.mathml()
                    + mtag('i', 'd')
                    + mtag('i', self.variable))

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return ' '.join((self.child.rpn(), self.variable, self.lower.rpn(), self.upper.rpn(), self.symbol))

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        indefinite = self.child.simplify().integral(self.variable)
        expanded = Subtraction(indefinite.substitute(self.variable, self.upper),
                               indefinite.substitute(self.variable, self.lower))
        return expanded.simplify()

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child.tuple(), self.variable, self.lower.tuple(), self.upper.tuple(), self.symbol

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()},' \
               f'{{{self.variable}, {self.lower.wolfram()}, {self.upper.wolfram()}}}]'


class Piecewise(Node):
    """Piecewise function node"""
    wolfram_func = 'Piecewise'
    symbol = 'piecewise'
    __slots__ = 'expressions', 'default'

    def __init__(self, expressions: List[Tuple[Node, Node]], default: Optional[Node] = None):
        super().__init__()
        self.default = default.copy() if default is not None else Constant(0)
        self.expressions = [(expr.copy(), cond.copy()) for expr, cond in expressions]

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return Piecewise([(expr, cond) for expr, cond in self.expressions], self.default)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Piecewise([(expr.derivative(variable), cond) for expr, cond in self.expressions],
                         self.default.derivative(variable))

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
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
        return Piecewise([(expr.integral(var), cond) for expr, cond in self.expressions], self.default.integral(var))

    def latex(self) -> str:
        """return latex language representation of the tree"""
        raise NotImplementedError('Latex representation of piecewise functions not supported')

    def list_nodes(self) -> List['Node']:
        """return latex language representation of the tree"""
        out = [self]  # type: List[Node]
        for expr, cond in self.expressions:
            out += expr.list_nodes()
            out += cond.list_nodes()
        return out + self.default.list_nodes()

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        expression_part = ''
        for expr, cond in self.expressions:
            expression_part += mtag('tr',
                                    mtag('td', expr.mathml())
                                    + mtag('td',
                                           mtag('text', '&#xa0;<!--NO-BREAK SPACE--> if &#xa0;<!--NO-BREAK SPACE-->'),
                                           'columnalign="left"')
                                    + mtag('td', cond.mathml()))
        expression_part += mtag('tr',
                                mtag('td', self.default.mathml())
                                + mtag('td',
                                       mtag('text', '&#xa0;<!--NO-BREAK SPACE--> otherwise'),
                                       'columnaling="left"'))
        return mtag('row',
                    mtag('o', '{')
                    + mtag('table',
                           expression_part))

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        raise NotImplementedError('Rpn notation of piecewise functions not supported')

    def simplify(self) -> 'Node':
        """returns a simplified version of the tree"""
        return Piecewise([(expr.simplify(), cond.simplify()) for expr, cond in self.expressions],
                         self.default.simplify())

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return Piecewise([(expr.substitute(var, sub), cond.substitute(var, sub)) for expr, cond in self.expressions],
                         self.default.substitute(var, sub))

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return (tuple((expr.tuple(), cond.tuple()) for expr, cond in self.expressions)
                + (self.default.tuple(), self.symbol))

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        expression_part = '{'
        for expr, cond in self.expressions:
            expression_part += f'{{{expr.infix()}, {cond.infix()}}}, '
        expression_part += '}, ' + self.default.infix()
        return f'{self.wolfram_func}[{expression_part}]'


if __name__ == '__main__':
    x = Variable('x')
    expr = sum((3 - i) * (x ** i) for i in range(3))
    print(expr)
    print(expr.simplify())

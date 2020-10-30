"""
basic expression tree with evaluation and derivation
"""

from abc import ABCMeta, abstractmethod
from itertools import combinations_with_replacement
from math import e, log, sin, cos, tan, asin, acos, atan, isclose
from typing import Optional, Dict, Union, Tuple, List, Set

Number = Union[int, float]
Variables = Dict[str, Number]
Expression = Union[Tuple[Union[tuple, Number, str], Union[tuple, Number, str], str],
                   Tuple[Union[tuple, Number, str], str],
                   Number,
                   str]


def tag(xml_tag: str, content: str):
    return f'<m{xml_tag}>{content}</m{xml_tag}>'


class Node(metaclass=ABCMeta):
    """Abstract Base Class for any node in the expression tree"""
    __slots__ = 'parent',

    def __init__(self) -> None:
        self.parent: Optional[Node] = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __str__(self) -> str:
        return self.rpn()

    def __eq__(self, other: 'Node') -> bool:
        assert isinstance(other, Node)
        # todo: un-brute force this
        full_tree = Subtraction(self, other)
        dependencies = full_tree.dependencies()
        return all(isclose(full_tree.evaluate({letter: nr for letter, nr in zip(dependencies, values)}), 0)
                   for values in combinations_with_replacement([-2 ** x for x in range(20, -21, -1)]
                                                               + [2 ** x for x in range(-20, 21)], len(dependencies)))

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
        """returns a list of all nodes in the tree"""

    @abstractmethod
    def mathml(self) -> str:
        """returns the MathML representation of the tree"""

    @abstractmethod
    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""

    # @abstractmethod
    # def simplify(self):
    #     """returns a simplified version of the tree"""
    #     # todo: implement

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
    __slots__ = ()

    def list_nodes(self) -> List[Node]:
        """returns a list of all nodes in the tree"""
        return [self]


class Constant(Term):
    """Real numerical constant in expression tree, """
    __slots__ = 'value',

    def __init__(self, value: Number) -> None:
        super().__init__()
        assert isinstance(value, (int, float))
        self.value = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        return Constant(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.value)

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        return Product(self, Variable(var))

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return str(self.value)

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return tag('row',
                   tag('n',
                       str(self.value)))

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return str(self.value)

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.copy()

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.value

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'Real[{self.value}]'


class Variable(Term):
    """Named variable in expression tree"""
    __slots__ = 'symbol',

    def __init__(self, symbol: str) -> None:
        super().__init__()
        assert isinstance(symbol, str)
        self.symbol = symbol

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.symbol})'

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.symbol)

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return {self.symbol}

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the (partial) derivative to the passed variable of this tree"""
        if self.symbol == variable:
            return Constant(1)
        return Constant(0)

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        if var_dict is None:
            raise KeyError(f'None does not contain "{self.symbol}"')
        return var_dict[self.symbol]

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return self.symbol

    def integral(self, var: str) -> 'Node':
        """returns an expression tree representing the antiderivative to the passed variable of this tree"""
        if self.symbol == var:
            if self.parent is None or isinstance(self.parent, (Addition, Subtraction)):
                return Division(Exponent(self, Constant(2)), Constant(2))
            else:
                return self
        else:
            return Constant(0)

    def latex(self) -> str:
        """return latex language representation of the tree"""
        return self.symbol

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        return tag('row',
                   tag('<i>',
                       str(self.symbol)))

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.symbol

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        if self.symbol == var:
            return sub.copy()
        return self.copy()

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.symbol

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return self.symbol


class Operator2In(Node, metaclass=ABCMeta):
    """Abstract Base Class for 2-input operator in expression tree"""
    __slots__ = 'child1', 'child2',
    symbol = ''
    wolfram_func = ''

    def __init__(self, child1: Node, child2: Node) -> None:
        super().__init__()
        assert isinstance(child1, Node)
        assert isinstance(child2, Node)
        self.child1 = child1.copy()
        self.child2 = child2.copy()
        self.reset_parents()

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
            return tag('row',
                       tag('fenced',
                           tag('row',
                               self.child1.mathml()
                               + tag('o', self.symbol)
                               + self.child2.mathml())))
        else:
            return tag('row',
                       self.child1.mathml()
                       + tag('o', self.symbol)
                       + self.child2.mathml())

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        self.child1.reset_parents(self)
        self.child2.reset_parents(self)

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.child1.rpn() + ' ' + self.child2.rpn() + ' ' + self.symbol

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child1.substitute(var, sub), self.child2.substitute(var, sub))

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child1.tuple(), self.child2.tuple(), self.symbol

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child1.wolfram()}, {self.child2.wolfram()}]'


class Addition(Operator2In):
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
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, Operator1In)) or (
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
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, Operator1In)) or (
                isinstance(self.parent, Subtraction) and self.parent.child1 is self):
            return f'{self.child1.latex()} {self.symbol} {self.child2.latex()}'
        else:
            return f'({self.child1.latex()} {self.symbol} {self.child2.latex()})'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, Operator1In)) or (
                isinstance(self.parent, Subtraction) and self.parent.child1 is self):
            return tag('row',
                       self.child1.mathml()
                       + tag('o', self.symbol)
                       + self.child2.mathml())
        else:
            return tag('row',
                       tag('fenced',
                           tag('row',
                               self.child1.mathml()
                               + tag('o', self.symbol)
                               + self.child2.mathml())))


class Subtraction(Operator2In):
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


class Product(Operator2In):
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
                    raise err
            except (ArithmeticError, ValueError) as err:
                ans2 = self.child2.evaluate(var_dict)
                if ans2 == 0:
                    return 0
                else:
                    raise err

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


class Division(Operator2In):
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
        return tag('row',
                   tag('frac',
                       self.child1.mathml()
                       + self.child2.mathml()))


class Exponent(Operator2In):
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
                    raise err
            except (ArithmeticError, ValueError) as err:
                ans2 = self.child2.evaluate(var_dict)
                if ans2 == 0:
                    return 1
                else:
                    raise err

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
        """returns infix representation of the tree"""
        if isinstance(self.parent, Exponent):
            return f'({self.child1.latex()} {self.symbol} {self.child2.latex()})'
        else:
            return f'{self.child1.latex()} {self.symbol} {self.child2.latex()}'

    def mathml(self) -> str:
        """returns the MathML representation of the tree"""
        if isinstance(self.parent, Exponent):
            return tag('row',
                       tag('fenced',
                           tag('row',
                               tag('sup',
                                   self.child1.mathml()
                                   + self.child2.mathml()))))
        else:
            return tag('row',
                       tag('sup',
                           self.child1.mathml()
                           + self.child2.mathml()))


class Logarithm(Operator2In):
    """Logarithm operator node, child 2 is base"""
    __slots__ = ()
    symbol = 'log'
    wolfram_func = 'Log'

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
        return tag('row',
                   tag('sub',
                       tag('i', self.symbol)
                       + self.child2.mathml())
                   + tag('fenced', self.child1.mathml()))


class Operator1In(Node, metaclass=ABCMeta):
    """Abstract Base Class for single-input operator in expression tree"""
    __slots__ = 'child',
    symbol = ''
    wolfram_func = ''
    latex_func = ''

    def __init__(self, child: Node) -> None:
        super().__init__()
        assert isinstance(child, Node)
        self.child = child.copy()
        self.reset_parents()

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
        return tag('row',
                   tag('i', self.symbol)
                   + tag('fenced', self.child.mathml()))

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        self.child.reset_parents(self)

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.child.rpn() + ' ' + self.symbol

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child.substitute(var, sub))

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child.tuple(), self.symbol

    def wolfram(self) -> str:
        """return wolfram language representation of the tree"""
        return f'{self.wolfram_func}[{self.child.wolfram()}]'


class Sine(Operator1In):
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


class Cosine(Operator1In):
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


class Tangent(Operator1In):
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


class ArcSine(Operator1In):
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


class ArcCosine(Operator1In):
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


class ArcTangent(Operator1In):
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


class Absolute(Operator1In):
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
        return tag('row',
                   tag('o', '|')
                   + self.child.mathml()
                   + tag('o', '|'))

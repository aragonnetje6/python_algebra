"""
basic expression tree with evaluation and derivation
"""

from abc import ABCMeta, abstractmethod
from math import e, pi, log, sin, cos, tan, asin, acos, atan
from typing import Optional, Dict, Union, Tuple, List, Set

Number = Union[int, float]
Variables = Dict[str, Number]
Expression = Union[Tuple[Union[tuple, Number, str], Union[tuple, Number, str], str],
                   Tuple[Union[tuple, Number, str], str],
                   Number,
                   str]


class Node(metaclass=ABCMeta):
    """Abstract Base Class for any node in the expression tree"""
    __slots__ = 'parent',

    def __init__(self) -> None:
        self.parent: Optional[Node] = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __str__(self) -> str:
        return self.rpn()

    @abstractmethod
    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""

    @abstractmethod
    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""

    @abstractmethod
    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""

    @abstractmethod
    def list_nodes(self) -> List['Node']:
        """returns a list of all nodes in the tree"""

    @abstractmethod
    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""

    @abstractmethod
    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""

    @abstractmethod
    def infix(self) -> str:
        """returns infix representation of the tree"""

    @abstractmethod
    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""

    def reset_parents(self, parent: Optional['Node'] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        self.parent = parent

    def get_root(self) -> 'Node':
        """Returns the root node of the expression tree"""
        if self.parent is None:
            return self
        return self.parent.get_root()

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

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__()

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return set()


class Term(Node, metaclass=ABCMeta):
    """Abstract Base Class for any value (leaf node) in the expression tree"""
    __slots__ = ()

    @abstractmethod
    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""

    @abstractmethod
    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""

    @abstractmethod
    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""

    @abstractmethod
    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""

    @abstractmethod
    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""

    @abstractmethod
    def infix(self) -> str:
        """returns infix representation of the tree"""

    @abstractmethod
    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""

    def list_nodes(self) -> List[Node]:
        """returns a list of all nodes in the tree"""
        return [self]


class Constant(Term):
    """Numerical constant in expression tree, """
    __slots__ = 'value',

    def __init__(self, value: Number) -> None:
        super().__init__()
        self.value = value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.copy()

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.value

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.value

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return str(self.value)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Constant(0)

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.value)

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        return True, ''

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return str(self.value)


class Variable(Term):
    """Named variable in expression tree"""
    __slots__ = 'symbol',

    def __init__(self, symbol: str) -> None:
        super().__init__()
        self.symbol = symbol

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.symbol})'

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        if self.symbol == var:
            return sub.copy()
        return self.copy()

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        if var_dict is None:
            raise KeyError(f'None does not contain "{self.symbol}"')
        return var_dict[self.symbol]

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.symbol

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.symbol

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        if self.symbol == variable:
            return Constant(1)
        return Constant(0)

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.symbol)

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return {self.symbol}

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        if var_dict is None:
            return False, 'No variables given'
        if self.symbol in var_dict.keys():
            return True, ''
        else:
            return False, f'Variable {self.symbol} not given'

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return self.symbol


class Operator2In(Node, metaclass=ABCMeta):
    """Abstract Base Class for 2-input operator in expression tree"""
    __slots__ = 'child1', 'child2',
    symbol = ''

    def __init__(self, child1: Node, child2: Node) -> None:
        super().__init__()
        self.child1 = child1.copy()
        self.child2 = child2.copy()
        self.reset_parents()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child1)}, {repr(self.child2)})'

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child1.substitute(var, sub), self.child2.substitute(var, sub))

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        self.child1.reset_parents(self)
        self.child2.reset_parents(self)

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child1.tuple(), self.child2.tuple(), self.symbol

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.child1.rpn() + ' ' + self.child2.rpn() + ' ' + self.symbol

    def list_nodes(self) -> List[Node]:
        """returns a list of all nodes in the tree"""
        out = [self]  # type: List[Node]
        return out + self.child1.list_nodes() + self.child2.list_nodes()

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.child1, self.child2)

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return self.child1.dependencies().union(self.child2.dependencies())

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        validation1 = self.child1.validate(var_dict)
        validation2 = self.child2.validate(var_dict)
        if not validation1[0]:
            return validation1
        elif not validation2[0]:
            return validation2
        else:
            return True, ''

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'({self.child1.infix()}{self.symbol}{self.child2.infix()})'


class Addition(Operator2In):
    """Addition operator node"""
    __slots__ = ()
    symbol = '+'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child1.evaluate(var_dict) + self.child2.evaluate(var_dict)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Addition(self.child1.derivative(variable), self.child2.derivative(variable))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, Operator1In)) or (
                isinstance(self.parent, Subtraction) and self.parent.child1 is self):
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'
        else:
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'


class Subtraction(Operator2In):
    """Subtraction operator node"""
    __slots__ = ()
    symbol = '-'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child1.evaluate(var_dict) - self.child2.evaluate(var_dict)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Subtraction(self.child1.derivative(variable), self.child2.derivative(variable))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if self.parent is None or isinstance(self.parent, (Addition, Logarithm, Operator1In)) or (
                isinstance(self.parent, Subtraction) and self.parent.child1 is self):
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'
        else:
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'


class Product(Operator2In):
    """Multiplication operator node"""
    __slots__ = ()
    symbol = '*'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        if ((self.child1.validate(var_dict)[0] and self.child1.evaluate(var_dict) == 0)
                or (self.child2.validate(var_dict)[0] and self.child2.evaluate(var_dict) == 0)):
            return 0
        return self.child1.evaluate(var_dict) * self.child2.evaluate(var_dict)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Addition(Product(self.child1, self.child2.derivative(variable)),
                        Product(self.child1.derivative(variable), self.child2))

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        child1_valid = self.child1.validate(var_dict)
        child2_valid = self.child2.validate(var_dict)
        if (child1_valid[0] and (child2_valid[0] or self.child1.evaluate(var_dict))) or \
                (child2_valid[0] and self.child2.evaluate(var_dict) == 0):
            return True, ''
        return next(filter(lambda x: not x[0], (child1_valid, child2_valid)))

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if (isinstance(self.parent, Division) and self.parent.child2 is self) or isinstance(self.parent, Exponent):
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'


class Division(Operator2In):
    """Division operator node"""
    __slots__ = ()
    symbol = '/'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return self.child1.evaluate(var_dict) / self.child2.evaluate(var_dict)

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Division(Subtraction(Product(self.child1.derivative(variable), self.child2),
                                    Product(self.child1, self.child2.derivative(variable))),
                        Exponent(self.child2, Constant(2)))

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        childtrees = super().validate(var_dict)
        if not childtrees[0]:
            return childtrees
        elif self.child2.evaluate(var_dict) == 0:
            return False, 'Division by zero'
        else:
            return True, ''

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if (isinstance(self.parent, Division) and self.parent.child2 is self) or isinstance(self.parent, Exponent):
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'


class Exponent(Operator2In):
    """Exponent operator node"""
    __slots__ = ()
    symbol = '^'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        child1_valid = self.child1.validate(var_dict)[0]
        child2_valid = self.child2.validate(var_dict)[0]
        if child1_valid:
            child1_result = self.child1.evaluate(var_dict)
            if child1_result == 0:
                return 0
            if child1_result == 1:
                return 1
        if child2_valid and self.child2.evaluate(var_dict) == 0:
            return 1
        result: Number = self.child1.evaluate(var_dict) ** self.child2.evaluate(var_dict)
        if isinstance(result, complex):
            raise ValueError('Complex values not allowed')
        return result

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Product(self,
                       Addition(Product(self.child1.derivative(variable),
                                        Division(self.child2,
                                                 self.child1)),
                                Product(self.child2.derivative(variable),
                                        Logarithm(self.child1,
                                                  Constant(e)))))

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        child1_valid = self.child1.validate(var_dict)
        child2_valid = self.child2.validate(var_dict)
        if child1_valid[0]:
            if child2_valid[0]:
                child1_result = self.child1.evaluate(var_dict)
                if child1_result in (0, 1):
                    return True, ''
                try:
                    if isinstance(child1_result ** self.child2.evaluate(var_dict), complex):
                        return False, 'Complex values not allowed'
                    else:
                        return True, ''
                except OverflowError:
                    return False, 'Overflow'
        if child2_valid[0] and self.child2.evaluate(var_dict) == 0:
            return True, ''
        return False, (child1_valid[1] if not child1_valid[0] else child2_valid[1])

    def infix(self) -> str:
        """returns infix representation of the tree"""
        if isinstance(self.parent, Exponent):
            return f'({self.child1.infix()} {self.symbol} {self.child2.infix()})'
        else:
            return f'{self.child1.infix()} {self.symbol} {self.child2.infix()}'


class Logarithm(Operator2In):
    """Logarithm operator node"""
    __slots__ = ()
    symbol = 'log'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return log(self.child1.evaluate(var_dict), self.child2.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Division(Subtraction(Division(Product(self.child1.derivative(variable),
                                                     Logarithm(self.child2,
                                                               Constant(e))),
                                             self.child1),
                                    Division(Product(self.child2.derivative(variable),
                                                     Logarithm(self.child1,
                                                               Constant(e)))
                                             , self.child2)),
                        Exponent(Logarithm(self.child2,
                                           Constant(e)),
                                 Constant(2)))

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        children_ok = super().validate(var_dict)
        if children_ok[0]:
            child1_result = self.child1.evaluate(var_dict)
            child2_result = self.child2.evaluate(var_dict)
            if child1_result <= 0:
                return False, 'Logarithm of negative number'
            elif 0 >= child2_result:
                return False, 'Logarithm with negative base'
            elif child2_result == 1:
                return False, 'Logarithm with base 1'
            else:
                return True, ''
        return children_ok

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'{self.symbol}({self.child1.infix()}, {self.child2.infix()})'


class Operator1In(Node, metaclass=ABCMeta):
    """Abstract Base Class for single-input operator in expression tree"""
    __slots__ = 'child',
    symbol = ''

    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child = child.copy()
        self.reset_parents()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.child)})'

    def substitute(self, var: str, sub: 'Node') -> 'Node':
        """substitute a variable with an expression inside this tree, returns the resulting tree"""
        return self.__class__(self.child.substitute(var, sub))

    def reset_parents(self, parent: Optional[Node] = None) -> None:
        """Resets the parent references of each descendant to the proper parent"""
        super().reset_parents(parent)
        self.child.reset_parents(self)

    def tuple(self) -> Expression:
        """returns the tuple representation of the tree"""
        return self.child.tuple(), self.symbol

    def rpn(self) -> str:
        """returns the reverse polish notation representation of the tree"""
        return self.child.rpn() + ' ' + self.symbol

    def list_nodes(self) -> List[Node]:
        """returns a list of all nodes in the tree"""
        out = [self]  # type: List[Node]
        return out + self.child.list_nodes()

    def copy(self) -> 'Node':
        """returns a copy of this tree"""
        return self.__class__(self.child)

    def dependencies(self) -> Set[str]:
        """returns set of all variables present in the tree"""
        return self.child.dependencies()

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        return self.child.validate(var_dict)

    def infix(self) -> str:
        """returns infix representation of the tree"""
        return f'{self.symbol}({self.child.infix()})'


class Sine(Operator1In):
    """Sine operator node in radians"""
    __slots__ = ()
    symbol = 'sin'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return sin(self.child.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Product(Cosine(self.child),
                       self.child.derivative(variable))


class Cosine(Operator1In):
    """Cosine operator node in radians"""
    __slots__ = ()
    symbol = 'cos'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return cos(self.child.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Subtraction(Constant(0),
                           Product(Cosine(self.child),
                                   self.child.derivative(variable)))


class Tangent(Operator1In):
    """Tangent operator node in radians"""
    __slots__ = ()
    symbol = 'tan'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return tan(self.child.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Cosine(self.child),
                                 Constant(2)))

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        child_ok = super().validate(var_dict)
        if child_ok[0]:
            if self.child.evaluate(var_dict) % pi == pi / 2:
                return False, 'Tangent of k*pi+pi/2 not defined'
            else:
                return True, ''
        return child_ok


class ArcSine(Operator1In):
    """Arcsine operator node in radians"""
    __slots__ = ()
    symbol = 'asin'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return asin(self.child.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Exponent(Subtraction(Constant(1),
                                             Exponent(self.child,
                                                      Constant(2))),
                                 Constant(1 / 2)))

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        child_ok = super().validate(var_dict)
        if child_ok[0]:
            if not -1 <= self.child.evaluate(var_dict) <= 1:
                return False, 'ArcSine must have value between -1 and 1'
            else:
                return True, ''
        return child_ok


class ArcCosine(Operator1In):
    """Arccosine operator node in radians"""
    __slots__ = ()
    symbol = 'acos'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return acos(self.child.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Subtraction(Constant(0),
                           Division(self.child.derivative(variable),
                                    Exponent(Subtraction(Constant(1),
                                                         Exponent(self.child,
                                                                  Constant(2))),
                                             Constant(1 / 2))))

    def validate(self, var_dict: Optional[Variables] = None) -> Tuple[bool, str]:
        """checks whether the tree represents a valid expression. returns a tuple with bool and str for reason"""
        child_ok = super().validate(var_dict)
        if child_ok[0]:
            if not -1 <= self.child.evaluate(var_dict) <= 1:
                return False, 'ArcCosine must have value between -1 and 1'
            else:
                return True, ''
        return child_ok


class ArcTangent(Operator1In):
    """Arctangent operator node in radians"""
    __slots__ = ()
    symbol = 'atan'

    def evaluate(self, var_dict: Optional[Variables] = None) -> Number:
        """Evaluates the expression tree using the values from var_dict, returns int or float"""
        return atan(self.child.evaluate(var_dict))

    def derivative(self, variable: str) -> 'Node':
        """returns an expression tree representing the derivative to the passed variable of this tree"""
        return Division(self.child.derivative(variable),
                        Addition(Constant(1),
                                 Exponent(self.child,
                                          Constant(2))))

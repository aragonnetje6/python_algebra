"""
Unittests for math_tree using pytest
"""

from hypothesis import given
from hypothesis.strategies import SearchStrategy, deferred, one_of, builds, sampled_from, booleans, integers, floats, \
    dictionaries
from math_tree import Node, Constant, Variable, Addition, Subtraction, Product, Division, Exponent, Logarithm, Equal, \
    NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor, Xnor, Sine, Cosine, Tangent, \
    ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert, Not, Derivative, IndefiniteIntegral, DefiniteIntegral, \
    Piecewise, Variables
from pytest import fixture


@fixture(scope="module")
def x() -> Variable:
    return Variable('x')


@fixture(scope="module")
def y() -> Variable:
    return Variable('y')


@fixture(scope="module")
def z() -> Variable:
    return Variable('z')


binary_operators = [Addition, Subtraction, Product, Division, Exponent, Logarithm]
unary_operators = [Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert]
binary_logical_operators = [Equal, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor,
                            Xnor]
unary_logical_operators = [Not]
calculus_operators = [Derivative, IndefiniteIntegral, DefiniteIntegral]
misc_operators = [Piecewise]


def variables_dict(keys: str, bools: bool = False) -> SearchStrategy[Variables]:
    if bools:
        return dictionaries(sampled_from(keys),
                            booleans(),
                            min_size=len(keys))
    else:
        return dictionaries(sampled_from(keys),
                            one_of(integers(), floats(allow_nan=False, allow_infinity=False)),
                            min_size=len(keys))


constant_number = builds(Constant, one_of(integers(), floats(allow_nan=False, allow_infinity=False)))
constant_bool = builds(Constant, booleans())
constant_any = one_of(constant_bool, constant_number)
variable = builds(Variable, sampled_from('xyz'))
func = lambda: (constant_number
                | variable
                | one_of(*[builds(operator, math_expression) for operator in unary_operators])
                | one_of(*[builds(operator, math_expression, math_expression)
                           for operator in binary_operators]))
math_expression = deferred(func)  # type: SearchStrategy[Node]


@given(val1=constant_any, val2=constant_any)
def test_equality(val1: Node, val2: Node) -> None:
    assert Equal(val1, val2).evaluate() == (val1.evaluate() == val2.evaluate())


@given(expr=math_expression, var_dict=variables_dict('xyz'))
def test_no_floats(expr: Node, var_dict: Variables) -> None:
    try:
        assert not isinstance(expr.evaluate(var_dict), float)
    except (OverflowError, ValueError, ArithmeticError):
        pass


class TestBinaryOperators:
    @given(var_dict=variables_dict('xy'))
    def test_1(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert Equal(x + y, y + x).evaluate(var_dict)

    @given(var_dict=variables_dict('xyz'))
    def test_2(self, x: Variable, y: Variable, z: Variable, var_dict: Variables) -> None:
        assert Equal((x + y) + z, x + (y + z)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_3(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x + 0, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_4(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x + -x, Constant(0)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_5(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert Equal(x + -y, x - y).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_6(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x - 0, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_7(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x + x, x * 2).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_8(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert Equal(x * y, y * x).evaluate(var_dict)

    @given(var_dict=variables_dict('xyz'))
    def test_9(self, x: Variable, y: Variable, z: Variable, var_dict: Variables) -> None:
        try:
            assert Equal((x * y) * z, x * (y * z)).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('xyz'))
    def test_10(self, x: Variable, y: Variable, z: Variable, var_dict: Variables) -> None:
        try:
            assert Equal(x * (y + z), x * y + x * z).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_11(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x * 1, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_12(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x * 0, Constant(0)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_13(self, x: Variable, var_dict: Variables) -> None:
        if x.evaluate(var_dict) != 0:
            assert Equal(x * Invert(x), Constant(1)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_14(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        if y.evaluate(var_dict) != 0:
            assert Equal(x * Invert(y), x / y).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_15(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x / 1, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_16(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert Equal(x * x, x ** 2).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_17(self, x: Variable, var_dict: Variables) -> None:
        assert (Equal(x ** 1, x) | Equal(x, Constant(0))).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_18(self, x: Variable, var_dict: Variables) -> None:
        assert Equal(x ** 0, Constant(1)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_19(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        try:
            assert Equal((x + y) ** 2, x ** 2 + y ** 2 + 2 * x * y).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_20(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert Equal(Logarithm(x, x), Constant(1)).evaluate(var_dict)
        except ValueError:
            assert x.evaluate(var_dict) == 0 or x.evaluate(var_dict) == 1


class TestLogicOperators:
    @given(var_dict=variables_dict('x', True))
    def test_not(self, x: Variable, var_dict: Variables):
        assert Equal(Not(x), Nand(x, x)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy', True))
    def test_and(self, x: Variable, y: Variable, var_dict: Variables):
        assert Equal(And(x, y), Not(Nand(x, y))).evaluate(var_dict)

    @given(var_dict=variables_dict('xy', True))
    def test_or(self, x: Variable, y: Variable, var_dict: Variables):
        assert Equal(Or(x, y), Nand(Not(x), Not(y))).evaluate(var_dict)

    @given(var_dict=variables_dict('xy', True))
    def test_xor(self, x: Variable, y: Variable, var_dict: Variables):
        assert Equal(Xor(x, y), And(Or(x, y), Nand(x, y))).evaluate(var_dict)

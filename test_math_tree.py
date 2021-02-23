"""
Unittests for math_tree using pytest
"""

from hypothesis import given
from hypothesis.strategies import SearchStrategy, deferred, one_of, builds, sampled_from, booleans, integers, floats, \
    dictionaries
from math_tree import Node, Constant, Variable, Sum, Subtraction, Product, Division, Exponent, Logarithm, \
    IsEqual, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor, Xnor, Sine, Cosine, \
    Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert, Not, Derivative, IndefiniteIntegral, \
    DefiniteIntegral, Piecewise, Variables
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


binary_operators = [Sum, Subtraction, Product, Division, Exponent, Logarithm]
unary_operators = [Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert]
binary_logical_operators = [IsEqual, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor,
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
    assert IsEqual(val1, val2).evaluate() == (val1.evaluate() == val2.evaluate())


class TestBinaryOperators:
    @given(var_dict=variables_dict('xy'))
    def test_1(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert IsEqual(x + y, y + x).evaluate(var_dict)

    @given(var_dict=variables_dict('xyz'))
    def test_2(self, x: Variable, y: Variable, z: Variable, var_dict: Variables) -> None:
        assert IsEqual((x + y) + z, x + (y + z)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_3(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x + 0, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_4(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x + -x, Constant(0)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_5(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert IsEqual(x + -y, x - y).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_6(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x - 0, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_7(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x + x, x * 2).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_8(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert IsEqual(x * y, y * x).evaluate(var_dict)

    @given(var_dict=variables_dict('xyz'))
    def test_9(self, x: Variable, y: Variable, z: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual((x * y) * z, x * (y * z)).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('xyz'))
    def test_10(self, x: Variable, y: Variable, z: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual(x * (y + z), x * y + x * z).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_11(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x * 1, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_12(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x * 0, Constant(0)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_13(self, x: Variable, var_dict: Variables) -> None:
        if x.evaluate(var_dict) != 0:
            assert IsEqual(x * Invert(x), Constant(1)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_14(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        try:
            if y.evaluate(var_dict) != 0:
                assert IsEqual(x * Invert(y), x / y).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_15(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x / 1, x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_16(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual(x * x, x ** 2).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_17(self, x: Variable, var_dict: Variables) -> None:
        assert (IsEqual(x ** 1, x) | IsEqual(x, Constant(0))).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_18(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x ** 0, Constant(1)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_19(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual((x + y) ** 2, x ** 2 + y ** 2 + 2 * x * y).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_20(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual(Logarithm(x, x), Constant(1)).evaluate(var_dict)
        except (ValueError, ZeroDivisionError):
            pass


class TestLogicOperators:
    @given(var_dict=variables_dict('x', True))
    def test_not(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(Not(x), Nand(x, x)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy', True))
    def test_and(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert IsEqual(And(x, y), Not(Nand(x, y))).evaluate(var_dict)

    @given(var_dict=variables_dict('xy', True))
    def test_or(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert IsEqual(Or(x, y), Nand(Not(x), Not(y))).evaluate(var_dict)

    @given(var_dict=variables_dict('xy', True))
    def test_xor(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        assert IsEqual(Xor(x, y), And(Or(x, y), Nand(x, y))).evaluate(var_dict)


class TestUnaryOperators:
    @given(var_dict=variables_dict('x'))
    def test_1(self, x: Variable, var_dict: Variables) -> None:
        assert GreaterEqual(Absolute(x), Constant(0)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_2(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(Absolute(x), Absolute(Absolute(x))).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_3(self, x: Variable, var_dict: Variables) -> None:
        assert Xnor(GreaterEqual(Negate(x), Constant(0)), LessEqual(x, Constant(0))).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_4(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(Negate(Negate(x)), x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_5(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(Absolute(Negate(x)), Absolute(x)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_6(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert Xnor(GreaterEqual(Absolute(Invert(x)), Constant(1)),
                        LessEqual(Absolute(x), Constant(1))).evaluate(var_dict)
        except ZeroDivisionError:
            assert x.evaluate(var_dict) == 0

    @given(var_dict=variables_dict('x'))
    def test_7(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert Xnor(GreaterEqual(Invert(x), Constant(0)),
                        GreaterEqual(x, Constant(0))).evaluate(var_dict)
        except ZeroDivisionError:
            assert x.evaluate(var_dict) == 0

    @given(var_dict=variables_dict('x'))
    def test_8(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual(Invert(Invert(x)), x).evaluate(var_dict)
        except ZeroDivisionError:
            assert x.evaluate(var_dict) == 0
        except OverflowError:
            pass

"""
Unittests for math_tree using pytest
"""

from hypothesis import given
from hypothesis.strategies import SearchStrategy, deferred, one_of, builds, sampled_from, booleans, integers, floats, \
    dictionaries
from math_tree import Node, Nodeify, Variable, Sum, Subtraction, Product, Division, Exponent, Logarithm, \
    IsEqual, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor, Xnor, Sine, Cosine, \
    Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert, Not, Derivative, IndefiniteIntegral, \
    DefiniteIntegral, Piecewise, Variables, UnaryOperator, CalculusOperator, ArbitraryOperator, Integer
from pytest import fixture


@fixture(scope="module")
def x() -> Variable:
    """variable x fixture"""
    return Variable('x')


@fixture(scope="module")
def y() -> Variable:
    """variable x fixture"""
    return Variable('y')


@fixture(scope="module")
def z() -> Variable:
    """variable x fixture"""
    return Variable('z')


binary_operators = [Sum, Subtraction, Product, Division, Exponent, Logarithm]
unary_operators = [Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert]
binary_logical_operators = [IsEqual, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor,
                            Xnor]
unary_logical_operators = [Not]
calculus_operators = [Derivative, IndefiniteIntegral, DefiniteIntegral]
misc_operators = [Piecewise]


def variables_dict(keys: str, use_booleans: bool = False) -> SearchStrategy[Variables]:
    """create variable dictionary with given keys and values chosen from either numbers or booleans"""
    if use_booleans:
        return dictionaries(sampled_from(keys),
                            booleans(),
                            min_size=len(keys))
    else:
        return dictionaries(sampled_from(keys),
                            one_of(integers(), floats(-1e308, 1e308, allow_nan=False, allow_infinity=False)),
                            min_size=len(keys))


constant_number = builds(Nodeify, one_of(integers(), floats(-1e308, 1e308, allow_nan=False, allow_infinity=False)))
constant_bool = builds(Nodeify, booleans())
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
        assert IsEqual(x + -x, Integer(0)).evaluate(var_dict)

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
        assert IsEqual(x * 0, Integer(0)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_13(self, x: Variable, var_dict: Variables) -> None:
        if x.evaluate(var_dict) != 0:
            assert IsEqual(x * Invert(x), Integer(1)).evaluate(var_dict)

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
        assert (IsEqual(x ** 1, x) | IsEqual(x, Integer(0))).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_18(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(x ** 0, Integer(1)).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_19(self, x: Variable, y: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual((x + y) ** 2, x ** 2 + y ** 2 + 2 * x * y).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_20(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert IsEqual(Logarithm(x, x), Integer(1)).evaluate(var_dict)
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
        assert GreaterEqual(Absolute(x), Integer(0)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_2(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(Absolute(x), Absolute(Absolute(x))).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_3(self, x: Variable, var_dict: Variables) -> None:
        assert Xnor(GreaterEqual(Negate(x), Integer(0)), LessEqual(x, Integer(0))).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_4(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(Negate(Negate(x)), x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_5(self, x: Variable, var_dict: Variables) -> None:
        assert IsEqual(Absolute(Negate(x)), Absolute(x)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_6(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert Xnor(GreaterEqual(Absolute(Invert(x)), Integer(1)),
                        LessEqual(Absolute(x), Integer(1))).evaluate(var_dict)
        except ZeroDivisionError:
            assert x.evaluate(var_dict) == 0

    @given(var_dict=variables_dict('x'))
    def test_7(self, x: Variable, var_dict: Variables) -> None:
        try:
            assert Xnor(GreaterEqual(Invert(x), Integer(0)),
                        GreaterEqual(x, Integer(0))).evaluate(var_dict)
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
        except AssertionError:
            ans = x.evaluate()
            assert ans > 1e308 if not isinstance(ans, complex) else ans.imag > 1e308 or ans.real > 1e308


class TestDisplayMethods:
    @given(expr=math_expression)
    def test_wolfram_type(self, expr: Node) -> None:
        assert isinstance(expr.wolfram(), str)

    @given(expr=math_expression)
    def test_wolfram_recursive(self, expr: Node) -> None:
        if isinstance(expr, (UnaryOperator, CalculusOperator)):
            assert expr.child.wolfram() in expr.wolfram()
        elif isinstance(expr, ArbitraryOperator):
            for child in expr.children:
                assert child.wolfram() in expr.wolfram()

    @given(expr=math_expression)
    def test_mathml_type(self, expr: Node) -> None:
        assert isinstance(expr.mathml(), str)

    @given(expr=math_expression)
    def test_mathml_recursive(self, expr: Node) -> None:
        if isinstance(expr, (UnaryOperator, CalculusOperator)):
            assert expr.child.mathml() in expr.mathml()
        elif isinstance(expr, ArbitraryOperator):
            for child in expr.children:
                assert child.copy().mathml() in expr.mathml()

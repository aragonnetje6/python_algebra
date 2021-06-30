"""
Unittests for math_tree using pytest
"""

from fractions import Fraction
from typing import Callable

from hypothesis import given
from hypothesis.strategies import booleans, builds, complex_numbers, deferred, dictionaries, floats, fractions, \
    integers, one_of, sampled_from, SearchStrategy
import pytest

from math_tree import Absolute, And, ArcCosine, ArcSine, ArcTangent, Boolean, Complex, Cosine, Derivative, Division, \
    E, Environment, EvaluationError, Exponent, GreaterEqual, GreaterThan, Integer, Invert, IsEqual, LessEqual, \
    LessThan, Logarithm, Nand, Negate, Node, Nodeify, Nor, Not, NotEqual, Or, Pi, Piecewise, Product, Rational, Real, \
    Sine, Subtraction, Sum, Tangent, Variable, Xnor, Xor


# # sets of classes
# n_ary_operators = [Sum, Subtraction, Product, Division]
# binary_operators = [Exponent, Logarithm]
# unary_operators = [Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert]
# comparison_operators: list[Callable[..., Node]] = [IsEqual, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual]
# logical_operators = [And, Or, Nand, Nor, Xor, Xnor]
# unary_logical_operators = [Not]
# calculus_operators = [Derivative]
# misc_operators = [Piecewise]


# # hypothesis strategies
# constant_number = builds(Nodeify,
#                          one_of(integers(int(-1e10), int(1e10)),
#                                 fractions(),
#                                 floats(-1e10, 1e10, allow_nan=False, allow_infinity=False)))
# constant_bool = builds(Nodeify, booleans())
# constant_any = one_of(constant_bool, constant_number)
# variable = builds(Variable, sampled_from('xyz'))
# func = lambda: (constant_number
#                 | one_of(builds(E), builds(Pi))
#                 | variable
#                 | one_of(*[builds(operator, math_expression) for operator in unary_operators])
#                 | one_of(*[builds(operator, math_expression, math_expression) for operator in binary_operators])
#                 | one_of(*[builds(operator, math_expression, math_expression, math_expression)
#                            for operator in n_ary_operators]))
# math_expression = deferred(func)  # type: SearchStrategy[Node]
# func2 = lambda: (constant_bool
#                  | variable
#                  | one_of(*[builds(operator, bool_expression) for operator in unary_logical_operators])
#                  | one_of(*[builds(operator, bool_expression, bool_expression) for operator in logical_operators]))
# bool_expression = deferred(func2)  # type: SearchStrategy[Node]
#
#
# @given(val1=constant_any, val2=constant_any)
# def test_equality(val1: Node, val2: Node) -> None:
#     assert IsEqual(val1, val2).evaluate() == (val1.evaluate() == val2.evaluate())


test_expressions: list[Node] = [(Variable('x') + 1) ** 0,
                                (Variable('x') + 1) ** 1,
                                (Variable('x') + 1) ** 2,
                                (Variable('x') + 1) ** 3,
                                (Variable('x') * 1) ** 0,
                                (Variable('x') * 1) ** 1,
                                (Variable('x') * 1) ** 2,
                                (Variable('x') * 1) ** 3,
                                (Variable('x') + Variable('y')) ** 0,
                                (Variable('x') + Variable('y')) ** 1,
                                (Variable('x') + Variable('y')) ** 2,
                                (Variable('x') + Variable('y')) ** 3,
                                (Variable('x') * Variable('y')) ** 0,
                                (Variable('x') * Variable('y')) ** 1,
                                (Variable('x') * Variable('y')) ** 2,
                                (Variable('x') * Variable('y')) ** 3]


def environment(keys: str, use_booleans: bool = False, use_floats: bool = False) -> SearchStrategy[Environment]:
    """create variable dictionary with given keys and values chosen from either numbers or booleans"""
    if use_booleans:
        return dictionaries(sampled_from(keys),
                            booleans(),
                            min_size=len(keys))
    elif use_floats:
        return dictionaries(sampled_from(keys),
                            one_of(integers(int(-1000), int(1000)),
                                   floats(-1e10, 1e10, allow_nan=False, allow_infinity=False)),
                            min_size=len(keys))
    else:
        return dictionaries(sampled_from(keys),
                            integers(int(-1000), int(1000)),
                            min_size=len(keys))


class TestBinaryOperators:
    @given(env=environment('xy'))
    def test_1(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        assert IsEqual(x + y, y + x).evaluate(env)

    @given(env=environment('xyz'))
    def test_2(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        assert IsEqual((x + y) + z, x + (y + z)).evaluate(env)

    @given(env=environment('x'))
    def test_3(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x + 0, x).evaluate(env)

    @given(env=environment('x'))
    def test_4(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x + -x, Integer(0)).evaluate(env)

    @given(env=environment('xy'))
    def test_5(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        assert IsEqual(x + -y, x - y).evaluate(env)

    @given(env=environment('x'))
    def test_6(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x - 0, x).evaluate(env)

    @given(env=environment('x'))
    def test_7(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x + x, x * 2).evaluate(env)

    @given(env=environment('xy'))
    def test_8(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        assert IsEqual(x * y, y * x).evaluate(env)

    @given(env=environment('xyz'))
    def test_9(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        try:
            assert IsEqual((x * y) * z, x * (y * z)).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('xyz'))
    def test_10(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        try:
            assert IsEqual(x * (y + z), x * y + x * z).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_11(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x * 1, x).evaluate(env)

    @given(env=environment('x'))
    def test_12(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x * 0, Integer(0)).evaluate(env)

    @given(env=environment('x'))
    def test_13(self, env: Environment) -> None:
        x = Variable('x')
        if x.evaluate(env) != 0:
            assert IsEqual(x * Invert(x), Integer(1)).evaluate(env)

    @given(env=environment('xy'))
    def test_14(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        try:
            if y.evaluate(env) != 0:
                assert IsEqual(x * Invert(y), x / y).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_15(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x / 1, x).evaluate(env)

    @given(env=environment('x'))
    def test_16(self, env: Environment) -> None:
        x = Variable('x')
        try:
            assert IsEqual(x * x, x ** 2).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_17(self, env: Environment) -> None:
        x = Variable('x')
        assert (IsEqual(x ** 1, x) | IsEqual(x, Integer(0))).evaluate(env)

    @given(env=environment('x'))
    def test_18(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(x ** 0, Integer(1)).evaluate(env)

    @given(env=environment('xy'))
    def test_19(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        try:
            assert IsEqual((x + y) ** 2, x ** 2 + y ** 2 + 2 * x * y).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_20(self, env: Environment) -> None:
        x = Variable('x')
        try:
            assert IsEqual(Logarithm(x, x), Integer(1)).evaluate(env)
        except EvaluationError:
            pass


class TestLogicOperators:
    @given(env=environment('x', True))
    def test_not(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(Not(x), Nand(x, x)).evaluate(env)

    @given(env=environment('xy', True))
    def test_and(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        assert IsEqual(And(x, y), Not(Nand(x, y))).evaluate(env)

    @given(env=environment('xy', True))
    def test_or(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        assert IsEqual(Or(x, y), Nand(Not(x), Not(y))).evaluate(env)

    @given(env=environment('xy', True))
    def test_xor(self, env: Environment) -> None:
        x = Variable('x')
        y = Variable('y')
        assert IsEqual(Xor(x, y), And(Or(x, y), Nand(x, y))).evaluate(env)


class TestUnaryOperators:
    @given(env=environment('x'))
    def test_1(self, env: Environment) -> None:
        x = Variable('x')
        assert GreaterEqual(Absolute(x), Integer(0)).evaluate(env)

    @given(env=environment('x'))
    def test_2(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(Absolute(x), Absolute(Absolute(x))).evaluate(env)

    @given(env=environment('x'))
    def test_3(self, env: Environment) -> None:
        x = Variable('x')
        assert Xnor(GreaterEqual(Negate(x), Integer(0)), LessEqual(x, Integer(0))).evaluate(env)

    @given(env=environment('x'))
    def test_4(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(Negate(Negate(x)), x).evaluate(env)

    @given(env=environment('x'))
    def test_5(self, env: Environment) -> None:
        x = Variable('x')
        assert IsEqual(Absolute(Negate(x)), Absolute(x)).evaluate(env)

    @given(env=environment('x'))
    def test_6(self, env: Environment) -> None:
        x = Variable('x')
        try:
            assert Xnor(GreaterEqual(Absolute(Invert(x)), Integer(1)),
                        LessEqual(Absolute(x), Integer(1))).evaluate(env)
        except EvaluationError:
            assert x.evaluate(env) == 0

    @given(env=environment('x'))
    def test_7(self, env: Environment) -> None:
        x = Variable('x')
        try:
            assert Xnor(GreaterEqual(Invert(x), Integer(0)),
                        GreaterEqual(x, Integer(0))).evaluate(env)
        except EvaluationError:
            assert x.evaluate(env) == 0


class TestSimplifyTerms:
    @given(x=integers())
    def test_integer(self, x: int) -> None:
        value = Nodeify(x)
        simplified = value.simplify()
        assert isinstance(simplified, Integer)
        assert simplified.evaluate() == x
        assert simplified == simplified.simplify()

    @given(x=booleans())
    def test_bool(self, x: bool) -> None:
        value = Nodeify(x)
        simplified = value.simplify()
        assert isinstance(simplified, Boolean)
        assert simplified.evaluate() == x
        assert simplified == simplified.simplify()

    @given(x=floats(allow_nan=False, allow_infinity=False))
    def test_float(self, x: float) -> None:
        value = Nodeify(x)
        simplified = value.simplify()
        assert isinstance(simplified, (Real, Rational)) or (
                isinstance(simplified, Integer) and x.is_integer())
        assert simplified.evaluate() == x or simplified.evaluate() == Fraction(x)
        assert simplified == simplified.simplify()

    @given(x=complex_numbers(allow_infinity=False, allow_nan=False))
    def test_complex(self, x: complex) -> None:
        value = Nodeify(x)
        simplified = value.simplify()
        if x.imag != 0:
            assert isinstance(simplified, Complex)
        elif x.real.is_integer():
            assert isinstance(simplified, Integer)
        else:
            assert isinstance(simplified, (Real, Rational))
        assert simplified.evaluate() == x or simplified.evaluate() == Fraction(x.real)
        assert simplified == simplified.simplify()

    @given(env=environment('x'))
    def test_variable(self, env: Environment) -> None:
        value = Variable('x')
        simplified = value.simplify()
        simplified_alt = value.simplify(env)
        assert isinstance(simplified, Variable)
        assert value.name == simplified.name
        assert isinstance(simplified_alt, (Integer, Rational, Real, Complex))
        assert simplified_alt.evaluate() == value.evaluate(env)


class TestSimplifyCases:
    @given(env=environment('x'))
    def test_simple_sum(self, env: Environment) -> None:
        value = Sum(Variable('x'), Integer(1))
        simplified = value.simplify()
        assert simplified.evaluate(env) == value.evaluate(env)
        assert repr(simplified.simplify()) == repr(simplified)

    @given(env=environment('x'))
    def test_zero_sum(self, env: Environment) -> None:
        value = Sum(Variable('x'), Integer(0))
        simplified = value.simplify()
        assert simplified.evaluate(env) == value.evaluate(env)
        assert repr(simplified.simplify()) == repr(simplified)

    @given(env=environment('x'))
    def test_simple_product(self, env: Environment) -> None:
        value = Product(Variable('x'), Integer(2))
        simplified = value.simplify()
        assert simplified.evaluate(env) == value.evaluate(env)
        assert repr(simplified.simplify()) == repr(simplified)

    @given(env=environment('x'))
    def test_zero_product(self, env: Environment) -> None:
        value = Product(Variable('x'), Integer(0))
        simplified = value.simplify()
        assert simplified.evaluate(env) == value.evaluate(env)
        assert repr(simplified.simplify()) == repr(simplified)

    @given(env=environment('x'))
    def test_identity_product(self, env: Environment) -> None:
        value = Product(Variable('x'), Integer(1))
        simplified = value.simplify()
        assert simplified.evaluate(env) == value.evaluate(env)
        assert repr(simplified.simplify()) == repr(simplified)


@pytest.mark.parametrize('expression', test_expressions)
class TestSimplifyGeneral:
    def test_idempotence(self, expression: Node) -> None:
        simplified = expression.simplify()
        assert repr(simplified.simplify()) == repr(simplified)

    @given(env=environment('xy'))
    def test_same_answer(self, expression: Node, env: Environment) -> None:
        simplified = expression.simplify()
        assert expression.evaluate(env) == simplified.evaluate(env)

"""
Unittests for math_tree using pytest
"""

from fractions import Fraction
from typing import Callable

from hypothesis import given
from hypothesis.strategies import booleans, builds, complex_numbers, deferred, dictionaries, floats, fractions, \
    integers, one_of, sampled_from, SearchStrategy
from pytest import fixture

from math_tree import Absolute, And, ArcCosine, ArcSine, ArcTangent, Boolean, Complex, Cosine, Derivative, Division, E, \
    Environment, EvaluationError, Exponent, GreaterEqual, GreaterThan, Integer, Invert, IsEqual, LessEqual, LessThan, \
    Logarithm, Nand, Negate, Node, Nodeify, Nor, Not, NotEqual, Or, Pi, Piecewise, Product, Rational, Real, Sine, \
    Subtraction, Sum, Tangent, Variable, Xnor, Xor


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


n_ary_operators = [Sum, Subtraction, Product, Division]
binary_operators = [Exponent, Logarithm]
unary_operators = [Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert]
comparison_operators: list[Callable[..., Node]] = [IsEqual, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual]
logical_operators = [And, Or, Nand, Nor, Xor, Xnor]
unary_logical_operators = [Not]
calculus_operators = [Derivative]
misc_operators = [Piecewise]


def environment(keys: str, use_booleans: bool = False) -> SearchStrategy[Environment]:
    """create variable dictionary with given keys and values chosen from either numbers or booleans"""
    if use_booleans:
        return dictionaries(sampled_from(keys),
                            booleans(),
                            min_size=len(keys))
    else:
        return dictionaries(sampled_from(keys),
                            one_of(integers(int(-1e10), int(1e10)),
                                   floats(-1e10, 1e10, allow_nan=False, allow_infinity=False)),
                            min_size=len(keys))


constant_number = builds(Nodeify,
                         one_of(integers(int(-1e10), int(1e10)),
                                fractions(),
                                floats(-1e10, 1e10, allow_nan=False, allow_infinity=False)))
constant_bool = builds(Nodeify, booleans())
constant_any = one_of(constant_bool, constant_number)
variable = builds(Variable, sampled_from('xyz'))
func = lambda: (constant_number
                | one_of(builds(E), builds(Pi))
                | variable
                | one_of(*[builds(operator, math_expression) for operator in unary_operators])
                | one_of(*[builds(operator, math_expression, math_expression) for operator in binary_operators])
                | one_of(*[builds(operator, math_expression, math_expression, math_expression)
                           for operator in n_ary_operators]))
math_expression = deferred(func)  # type: SearchStrategy[Node]
func2 = lambda: (constant_bool
                 | variable
                 | one_of(*[builds(operator, bool_expression) for operator in unary_logical_operators])
                 | one_of(*[builds(operator, bool_expression, bool_expression) for operator in logical_operators]))
bool_expression = deferred(func2)  # type: SearchStrategy[Node]


@given(val1=constant_any, val2=constant_any)
def test_equality(val1: Node, val2: Node) -> None:
    assert IsEqual(val1, val2).evaluate() == (val1.evaluate() == val2.evaluate())


class TestBinaryOperators:
    @given(env=environment('xy'))
    def test_1(self, x: Variable, y: Variable, env: Environment) -> None:
        assert IsEqual(x + y, y + x).evaluate(env)

    @given(env=environment('xyz'))
    def test_2(self, x: Variable, y: Variable, z: Variable, env: Environment) -> None:
        assert IsEqual((x + y) + z, x + (y + z)).evaluate(env)

    @given(env=environment('x'))
    def test_3(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x + 0, x).evaluate(env)

    @given(env=environment('x'))
    def test_4(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x + -x, Integer(0)).evaluate(env)

    @given(env=environment('xy'))
    def test_5(self, x: Variable, y: Variable, env: Environment) -> None:
        assert IsEqual(x + -y, x - y).evaluate(env)

    @given(env=environment('x'))
    def test_6(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x - 0, x).evaluate(env)

    @given(env=environment('x'))
    def test_7(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x + x, x * 2).evaluate(env)

    @given(env=environment('xy'))
    def test_8(self, x: Variable, y: Variable, env: Environment) -> None:
        assert IsEqual(x * y, y * x).evaluate(env)

    @given(env=environment('xyz'))
    def test_9(self, x: Variable, y: Variable, z: Variable, env: Environment) -> None:
        try:
            assert IsEqual((x * y) * z, x * (y * z)).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('xyz'))
    def test_10(self, x: Variable, y: Variable, z: Variable, env: Environment) -> None:
        try:
            assert IsEqual(x * (y + z), x * y + x * z).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_11(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x * 1, x).evaluate(env)

    @given(env=environment('x'))
    def test_12(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x * 0, Integer(0)).evaluate(env)

    @given(env=environment('x'))
    def test_13(self, x: Variable, env: Environment) -> None:
        if x.evaluate(env) != 0:
            assert IsEqual(x * Invert(x), Integer(1)).evaluate(env)

    @given(env=environment('xy'))
    def test_14(self, x: Variable, y: Variable, env: Environment) -> None:
        try:
            if y.evaluate(env) != 0:
                assert IsEqual(x * Invert(y), x / y).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_15(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x / 1, x).evaluate(env)

    @given(env=environment('x'))
    def test_16(self, x: Variable, env: Environment) -> None:
        try:
            assert IsEqual(x * x, x ** 2).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_17(self, x: Variable, env: Environment) -> None:
        assert (IsEqual(x ** 1, x) | IsEqual(x, Integer(0))).evaluate(env)

    @given(env=environment('x'))
    def test_18(self, x: Variable, env: Environment) -> None:
        assert IsEqual(x ** 0, Integer(1)).evaluate(env)

    @given(env=environment('xy'))
    def test_19(self, x: Variable, y: Variable, env: Environment) -> None:
        try:
            assert IsEqual((x + y) ** 2, x ** 2 + y ** 2 + 2 * x * y).evaluate(env)
        except EvaluationError:
            pass

    @given(env=environment('x'))
    def test_20(self, x: Variable, env: Environment) -> None:
        try:
            assert IsEqual(Logarithm(x, x), Integer(1)).evaluate(env)
        except EvaluationError:
            pass


class TestLogicOperators:
    @given(env=environment('x', True))
    def test_not(self, x: Variable, env: Environment) -> None:
        assert IsEqual(Not(x), Nand(x, x)).evaluate(env)

    @given(env=environment('xy', True))
    def test_and(self, x: Variable, y: Variable, env: Environment) -> None:
        assert IsEqual(And(x, y), Not(Nand(x, y))).evaluate(env)

    @given(env=environment('xy', True))
    def test_or(self, x: Variable, y: Variable, env: Environment) -> None:
        assert IsEqual(Or(x, y), Nand(Not(x), Not(y))).evaluate(env)

    @given(env=environment('xy', True))
    def test_xor(self, x: Variable, y: Variable, env: Environment) -> None:
        assert IsEqual(Xor(x, y), And(Or(x, y), Nand(x, y))).evaluate(env)


class TestUnaryOperators:
    @given(env=environment('x'))
    def test_1(self, x: Variable, env: Environment) -> None:
        assert GreaterEqual(Absolute(x), Integer(0)).evaluate(env)

    @given(env=environment('x'))
    def test_2(self, x: Variable, env: Environment) -> None:
        assert IsEqual(Absolute(x), Absolute(Absolute(x))).evaluate(env)

    @given(env=environment('x'))
    def test_3(self, x: Variable, env: Environment) -> None:
        assert Xnor(GreaterEqual(Negate(x), Integer(0)), LessEqual(x, Integer(0))).evaluate(env)

    @given(env=environment('x'))
    def test_4(self, x: Variable, env: Environment) -> None:
        assert IsEqual(Negate(Negate(x)), x).evaluate(env)

    @given(env=environment('x'))
    def test_5(self, x: Variable, env: Environment) -> None:
        assert IsEqual(Absolute(Negate(x)), Absolute(x)).evaluate(env)

    @given(env=environment('x'))
    def test_6(self, x: Variable, env: Environment) -> None:
        try:
            assert Xnor(GreaterEqual(Absolute(Invert(x)), Integer(1)),
                        LessEqual(Absolute(x), Integer(1))).evaluate(env)
        except EvaluationError:
            assert x.evaluate(env) == 0

    @given(env=environment('x'))
    def test_7(self, x: Variable, env: Environment) -> None:
        try:
            assert Xnor(GreaterEqual(Invert(x), Integer(0)),
                        GreaterEqual(x, Integer(0))).evaluate(env)
        except EvaluationError:
            assert x.evaluate(env) == 0


class TestSimplify:
    @given(x=integers())
    def test_integer(self, x: int) -> None:
        value = Nodeify(x)
        assert isinstance(value.simplify(), Integer)
        assert value.simplify().evaluate() == x
        assert value.simplify() == value.simplify().simplify()

    @given(x=booleans())
    def test_bool(self, x: bool) -> None:
        value = Nodeify(x)
        assert isinstance(value.simplify(), Boolean)
        assert value.simplify().evaluate() == x
        assert value.simplify() == value.simplify().simplify()

    @given(x=floats(allow_nan=False, allow_infinity=False))
    def test_float(self, x: float) -> None:
        value = Nodeify(x)
        assert isinstance(value.simplify(), (Real, Rational)) or (
                    isinstance(value.simplify(), Integer) and x.is_integer())
        assert value.simplify().evaluate() == x or value.simplify().evaluate() == Fraction(x)
        assert value.simplify() == value.simplify().simplify()

    @given(x=complex_numbers(allow_infinity=False, allow_nan=False))
    def test_complex(self, x: complex) -> None:
        value = Nodeify(x)
        if x.imag != 0:
            assert isinstance(value.simplify(), Complex)
        elif x.real.is_integer():
            assert isinstance(value.simplify(), Integer)
        else:
            assert isinstance(value.simplify(), (Real, Rational))
        assert value.simplify().evaluate() == x or value.simplify().evaluate() == Fraction(x.real)
        assert value.simplify() == value.simplify().simplify()


# todo: add specific case tests for simplification rules
# class TestSimplify:
#     @settings(deadline=1000)
#     @given(env=environment('xyz'), expr=math_expression)
#     def test_same_answer(self, expr: Node, env: Environment) -> None:
#         try:
#             assert IsEqual(expr, expr.simplify()).evaluate(env)
#         except EvaluationError:
#             with raises(EvaluationError):
#                 expr.evaluate(env)
#
#     @settings(deadline=1000)
#     @given(expr=math_expression)
#     def test_idempotence(self, expr: Node) -> None:
#         assert repr(a := expr.simplify()) == repr(a.simplify())
#
#     @settings(deadline=1000)
#     @given(env=environment('xyz', use_booleans=True), expr=bool_expression)
#     def test_same_answer_bool(self, expr: Node, env: Environment) -> None:
#         try:
#             assert expr.evaluate(env) == expr.simplify().evaluate(env)
#         except EvaluationError:
#             with raises(EvaluationError):
#                 expr.evaluate(env)
#
#     @settings(deadline=1000)
#     @given(expr=bool_expression)
#     def test_idempotence_bool(self, expr: Node) -> None:
#         assert repr(a := expr.simplify()) == repr(a.simplify())
#
#
# class TestDisplayMethods:
#     @given(expr=math_expression)
#     def test_wolfram_type(self, expr: Node) -> None:
#         assert isinstance(expr.wolfram(), str)
#
#     @given(expr=math_expression)
#     def test_wolfram_recursive(self, expr: Node) -> None:
#         if isinstance(expr, (UnaryOperator, Derivative)):
#             assert expr.child.wolfram() in expr.wolfram()
#         elif isinstance(expr, ArbitraryOperator):
#             for child in expr.children:
#                 assert child.wolfram() in expr.wolfram()
#
#     @given(expr=math_expression)
#     def test_mathml_type(self, expr: Node) -> None:
#         assert isinstance(expr.mathml(), str)
#
#     @given(expr=math_expression)
#     def test_mathml_recursive(self, expr: Node) -> None:
#         if isinstance(expr, (UnaryOperator, Derivative)):
#             assert expr.child.mathml() in expr.mathml()
#         elif isinstance(expr, ArbitraryOperator):
#             for child in expr.children:
#                 assert child.mathml() in expr.mathml()

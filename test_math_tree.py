"""
Unittests for math_tree using pytest
"""

from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy, deferred, one_of, builds, sampled_from, booleans, integers, floats, \
    dictionaries
from math_tree import Node, Constant, Variable, Addition, Subtraction, Product, Division, Exponent, Logarithm, Equal, \
    NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor, Xnor, Sine, Cosine, Tangent, \
    ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert, Not, Derivative, IndefiniteIntegral, DefiniteIntegral, \
    Piecewise

x = Variable('x')
y = Variable('y')
z = Variable('z')

binary_operators = [Addition, Subtraction, Product, Division, Exponent, Logarithm]
unary_operators = [Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert]
binary_logical_operators = [Equal, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor,
                            Xnor]
unary_logical_operators = [Not]
calculus_operators = [Derivative, IndefiniteIntegral, DefiniteIntegral]
misc_operators = [Piecewise]


def variables_dict(keys: str, bools: bool = False) -> SearchStrategy:
    if bools:
        return dictionaries(sampled_from(keys), booleans())
    else:
        return dictionaries(sampled_from(keys),
                            one_of(integers(), floats(allow_nan=False, allow_infinity=False)),
                            min_size=len(keys))


constant_number = builds(Constant, one_of(integers(), floats(allow_nan=False, allow_infinity=False)))
constant_bool = builds(Constant, booleans())
constant_any = one_of(constant_bool, constant_number)
variable = builds(Variable, sampled_from('xy'))
math_expression = deferred(lambda: constant_number
                                   | variable
                                   | one_of(*[builds(operator, math_expression) for operator in unary_operators])
                                   | one_of(*[builds(operator, math_expression, math_expression)
                                              for operator in binary_operators]))


@given(val1=constant_any, val2=constant_any)
def test_equality(val1: Node, val2: Node):
    assert (val1 == val2).evaluate() == (val1.evaluate() == val2.evaluate())


class TestIdentities:
    @given(var_dict=variables_dict('xy'))
    def test_1(self, var_dict):
        assert (x + y == y + x).evaluate(var_dict)

    @given(var_dict=variables_dict('xyz'))
    def test_2(self, var_dict):
        assert ((x + y) + z == x + (y + z)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_3(self, var_dict):
        assert (x + 0 == x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_4(self, var_dict):
        assert (x + -x == 0).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_5(self, var_dict):
        assert (x + -y == x - y).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_6(self, var_dict):
        assert (x - 0 == x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_7(self, var_dict):
        assert (x + x == x * 2).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_8(self, var_dict):
        assert (x * y == y * x).evaluate(var_dict)

    @given(var_dict=variables_dict('xyz'))
    def test_9(self, var_dict):
        try:
            assert ((x * y) * z == x * (y * z)).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('xyz'))
    def test_10(self, var_dict):
        try:
            assert (x * (y + z) == x * y + x * z).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_11(self, var_dict):
        assert (x * 1 == x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_12(self, var_dict):
        assert (x * 0 == 0).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_13(self, var_dict):
        if x.evaluate(var_dict) != 0:
            assert (x * Invert(x) == 1).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_14(self, var_dict):
        if y.evaluate(var_dict) != 0:
            assert (x * Invert(y) == x / y).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_15(self, var_dict):
        assert (x / 1 == x).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_16(self, var_dict):
        try:
            assert (x * x == x ** 2).evaluate(var_dict)
        except OverflowError:
            pass

    @given(var_dict=variables_dict('x'))
    def test_17(self, var_dict):
        assert ((x ** 1 == x) | (x == 0)).evaluate(var_dict)

    @given(var_dict=variables_dict('x'))
    def test_18(self, var_dict):
        assert (x ** 0 == 1).evaluate(var_dict)

    @given(var_dict=variables_dict('xy'))
    def test_19(self, var_dict):
        try:
            assert ((x + y) ** 2 == x ** 2 + y ** 2 + 2 * x * y).evaluate(var_dict)
        except OverflowError:
            pass

    # @given(var_dict=variables_dict('xy'))
    # def test_20(self, var_dict):
    #     try:
    #         assert (Logarithm(x ** y, x) == y).evaluate(var_dict)
    #     except (ValueError, ArithmeticError):
    #         assert x.evaluate(var_dict) <= 0 or x.evaluate(var_dict) == 1
    #     except OverflowError:
    #         pass

    @given(var_dict=variables_dict('x'))
    def test_21(self, var_dict):
        try:
            assert (Logarithm(x, x) == 1).evaluate(var_dict)
        except (ValueError, ArithmeticError):
            assert x.evaluate(var_dict) <= 0 or x.evaluate(var_dict) == 1

# class TestTransformation:
#     @settings(deadline=10000)
#     @given(expr=math_expression)
#     def test_simplify(self, expr: Node):
#         try:
#             assert expr.evaluate() == expr.simplify().evaluate()
#         except (ValueError, ArithmeticError):
#             pass
#
#     @settings(deadline=10000)
#     @given(expr=math_expression)
#     def test_polynomial(self, expr: Node):
#         try:
#             assert expr.evaluate() == expr.polynomial().evaluate()
#         except (ValueError, ArithmeticError):
#             pass

"""
Unittests for math_tree using pytest
"""

from hypothesis import given
from hypothesis.strategies import deferred, one_of, builds, sampled_from, booleans, integers, floats, dictionaries
from math_tree import Node, Constant, Variable, Addition, Subtraction, Product, Division, Exponent, Logarithm, Equal, \
    NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor, Xnor, Sine, Cosine, Tangent, \
    ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert, Not, Derivative, IndefiniteIntegral, DefiniteIntegral

x = Variable('x')
y = Variable('y')
z = Variable('z')

binary_operators = [Addition, Subtraction, Product, Division, Exponent, Logarithm]
binary_logical_operators = [Equal, NotEqual, GreaterThan, LessThan, GreaterEqual, LessEqual, And, Or, Nand, Nor, Xor,
                            Xnor]
unary_operators = [Sine, Cosine, Tangent, ArcSine, ArcCosine, ArcTangent, Absolute, Negate, Invert]
unary_logical_operators = [Not]
calculus_operators = [Derivative, IndefiniteIntegral, DefiniteIntegral]

variables_dicts = dictionaries(sampled_from('abcdefghijklmnopqrstuvwxyz'),
                               one_of(integers(), floats(allow_nan=False, allow_infinity=False)),
                               min_size=26)

constant = builds(Constant, one_of(integers(), floats(allow_nan=False, allow_infinity=False)))
constant_bool = builds(Constant, booleans())
variable = builds(Variable, sampled_from('xy'))
expression = deferred(lambda: constant
                              | variable
                              | one_of(*[builds(operator, expression) for operator in unary_operators])
                              | one_of(*[builds(operator, expression, expression) for operator in binary_operators]))


class TestAlgebraProperties:
    """Property based testing for algebraic operators"""

    class TestAddition:
        def test_commutative(self):
            assert (x + y == y + x).evaluate()

        def test_associative(self):
            assert ((x + y) + z == x + (y + z)).evaluate()

        def test_identity(self):
            assert (x + 0 == x).evaluate()

        def test_negation(self):
            assert (x + -x == 0).evaluate()

    class TestSubtraction:
        def test_definition(self):
            assert (x + -y == x - y).evaluate()

        def test_identity(self):
            assert (x - 0 == x).evaluate()

    class TestProduct:
        def test_definition(self):
            assert (x + x == x * 2).evaluate()

        def test_commutative(self):
            assert (x * y == y * x).evaluate()

        def test_associative(self):
            assert ((x * y) * z == x * (y * z)).evaluate()

        def test_distributive(self):
            assert (x * (y + z) == x * y + x * z).evaluate()

        def test_identity(self):
            assert (x * 1 == x).evaluate()

        def test_annihilation(self):
            assert (x * 0 == 0).evaluate()

        def test_inversion(self):
            assert ((x * Invert(x) == 1) | (x == 0)).evaluate()

    class TestDivision:
        def test_definition(self):
            assert ((x * Invert(y) == x / y) | (x == 0)).evaluate()

        def test_identity(self):
            assert ((x / 1 == x) | (x == 0)).evaluate()

    class TestExponent:
        def test_definition(self):
            assert (x * x == x ** 2).evaluate()

        def test_identity(self):
            assert (x ** 1 == x).evaluate()

        def test_annihilation(self):
            assert ((x ** 0 == 1) | (x == 0)).evaluate()

        def test_distribution_over_addition(self):
            assert ((x + y) ** 2 == x ** 2 + y ** 2 + 2 * x * y).evaluate()

    class TestLogarithm:
        def test_definition(self):
            assert ((Logarithm(x ^ y, x) == y) | (x <= 0)).evaluate()

        def test_one(self):
            assert ((Logarithm(x, x) == 1) | (x <= 0)).evaluate()


class TestTransformation:
    @given(expr=expression)
    def test_simplify(self, expr: Node):
        try:
            assert (expr == expr.simplify()).evaluate()
        except (ValueError, ArithmeticError):
            pass

    @given(expr=expression)
    def test_polynomial(self, expr: Node):
        try:
            assert (expr == expr.polynomial()).evaluate()
        except (ValueError, ArithmeticError):
            pass

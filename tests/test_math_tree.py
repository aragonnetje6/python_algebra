"""
Unittests for math_tree using pytest
"""

import random
from string import ascii_lowercase

from python_algebra.math_tree import *

x = Variable('x')
y = Variable('y')
z = Variable('z')

var_dicts_ints = [{letter: random.randint(-100, 100) for letter in ascii_lowercase} for _ in range(100)]
var_dicts_floats = [{letter: random.random() * 200 - 100 for letter in ascii_lowercase} for _ in range(100)]


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

        def test_distribution_over_product(self):
            assert (((x * y) ** z == x ** z * y ** z) | (x < 0) | (y < 0)).evaluate()

    class TestLogarithm:
        def test_definition(self):
            assert ((Logarithm(x ^ y, x) == y) | (x <= 0)).evaluate()

        def test_one(self):
            assert ((Logarithm(x, x) == 1) | (x <= 0)).evaluate()

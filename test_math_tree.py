"""
Unittests for math_tree using pytest
"""

import random
from math import isclose

import pytest
from interpreter import rpn_to_tree
from math_tree import *

rpn_list = ['1 2 +',
            '0 0 +',
            '2 3 * 4 5 * +',
            '5 x 2 ^ * 4 x * + 3 +',
            '5 5 x + y 4 / + - 2 * 5 ^ x *',
            '5 5 x + y 4 + + + 2 * 5 + x log 5 x 2 ^ * 4 x * + 3 + log']

var_dicts_list = [{letter: random.randint(-100, 100) for letter in 'abcdefghijklmnopqrstuvwxyz'}]


class TestAddition:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Addition(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) + tree2.evaluate(variables),
                               abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(tree1.evaluate(variables) + tree2.evaluate(variables), 1., abs_tol=1e-09)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Addition(tree1, tree2)
        for var in total_tree.dependencies():
            tree1_derivative = tree1.derivative(var)
            tree2_derivative = tree2.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(ans_total, ans_der1 + ans_der2, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        isclose(tree1_derivative.evaluate(variables) + tree2_derivative.evaluate(variables), 1,
                                abs_tol=1e-09)


class TestSubtraction:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Subtraction(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) - tree2.evaluate(variables),
                               abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(tree1.evaluate(variables) - tree2.evaluate(variables), 1., abs_tol=1e-09)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Subtraction(tree1, tree2)
        for var in total_tree.dependencies():
            tree1_derivative = tree1.derivative(var)
            tree2_derivative = tree2.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(ans_total, ans_der1 - ans_der2, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        isclose(tree1_derivative.evaluate(variables) - tree2_derivative.evaluate(variables), 1,
                                abs_tol=1e-09)


class TestProduct:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Product(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) * tree2.evaluate(variables),
                               abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(tree1.evaluate(variables) * tree2.evaluate(variables), 1., abs_tol=1e-09)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Product(tree1, tree2)
        for var in total_tree.dependencies():
            tree1_derivative = tree1.derivative(var)
            tree2_derivative = tree2.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans1 = tree1.evaluate(variables)
                    ans2 = tree2.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(ans_total, ans_der1 * ans2 + ans1 * ans_der2, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        isclose(ans_der1 * ans2 + ans1 * ans_der2, 1., abs_tol=1e-09)


class TestDivision:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Division(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) / tree2.evaluate(variables),
                               abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(tree1.evaluate(variables) / tree2.evaluate(variables), 1., abs_tol=1e-09)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Division(tree1, tree2)
        for var in total_tree.dependencies():
            tree1_derivative = tree1.derivative(var)
            tree2_derivative = tree2.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans1 = tree1.evaluate(variables)
                    ans2 = tree2.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(ans_total, (ans_der1 * ans2 - ans1 * ans_der2) / ans2 ** 2, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        isclose((ans_der1 * ans2 + ans1 * ans_der2) / ans2 ** 2, 1., abs_tol=1e-09)


class TestExponent:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Exponent(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) ** tree2.evaluate(variables),
                               abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    if isinstance(tree1.evaluate(variables) ** tree2.evaluate(variables), complex):
                        raise ArithmeticError
                    isclose(tree1.evaluate(variables) ** tree2.evaluate(variables), 1., abs_tol=1e-09)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Exponent(tree1, tree2)
        for var in total_tree.dependencies():
            tree1_derivative = tree1.derivative(var)
            tree2_derivative = tree2.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans1 = tree1.evaluate(variables)
                    ans2 = tree2.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(ans_total, ans1 ** (ans2 - 1) * (ans2 * ans_der1 + ans1 * log(ans1) * ans_der2),
                                   abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        if isinstance(ans1 ** (ans2 - 1) * (ans2 * ans_der1 + ans1 * log(ans1) * ans_der2), complex):
                            raise ArithmeticError
                        isclose(ans1 ** (ans2 - 1) * (ans2 * ans_der1 + ans1 * log(ans1) * ans_der2), 1., abs_tol=1e-09)


class TestLogarithm:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Logarithm(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables),
                               log(tree1.evaluate(variables), tree2.evaluate(variables)), abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(log(tree1.evaluate(variables), tree2.evaluate(variables)), 1., abs_tol=1e-09)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Exponent(tree1, tree2)
        for var in total_tree.dependencies():
            tree1_derivative = tree1.derivative(var)
            tree2_derivative = tree2.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans1 = tree1.evaluate(variables)
                    ans2 = tree2.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(ans_total,
                                   ((ans_der1 * log(ans2) / ans1) - (ans_der2 * log(ans1) / ans2)) / log(ans2) ** 2,
                                   abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        isclose(((ans_der1 * log(ans2) / ans1) - (ans_der2 * log(ans1) / ans2)) / log(ans2) ** 2, 1.,
                                abs_tol=1e-09)


class TestSine:
    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_evaluate(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = Sine(input_tree)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), sin(input_tree.evaluate(variables)), abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(sin(input_tree.evaluate(variables)), 1., abs_tol=1e-09)

    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_derivative(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = Sine(input_tree)
        for var in total_tree.dependencies():
            input_tree_derivative = input_tree.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_input = input_tree.evaluate(variables)
                    ans_input_deriv = input_tree_derivative.evaluate(variables)
                    assert isclose(ans_total, cos(ans_input) * ans_input_deriv, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans_input = input_tree.evaluate(variables)
                        ans_input_deriv = input_tree_derivative.evaluate(variables)
                        isclose(cos(ans_input) * ans_input_deriv, 1., abs_tol=1e-09)


class TestCosine:
    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_evaluate(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = Cosine(input_tree)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), cos(input_tree.evaluate(variables)), abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(cos(input_tree.evaluate(variables)), 1., abs_tol=1e-09)

    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_derivative(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = Cosine(input_tree)
        for var in total_tree.dependencies():
            input_tree_derivative = input_tree.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_input = input_tree.evaluate(variables)
                    ans_input_deriv = input_tree_derivative.evaluate(variables)
                    assert isclose(ans_total, -sin(ans_input) * ans_input_deriv, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans_input = input_tree.evaluate(variables)
                        ans_input_deriv = input_tree_derivative.evaluate(variables)
                        isclose(-sin(ans_input) * ans_input_deriv, 1., abs_tol=1e-09)


class TestTangent:
    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_evaluate(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = Tangent(input_tree)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tan(input_tree.evaluate(variables)), abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(tan(input_tree.evaluate(variables)), 1., abs_tol=1e-09)

    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_derivative(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = Tangent(input_tree)
        for var in total_tree.dependencies():
            input_tree_derivative = input_tree.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_input = input_tree.evaluate(variables)
                    ans_input_deriv = input_tree_derivative.evaluate(variables)
                    assert isclose(ans_total, ans_input_deriv / cos(ans_input) ** 2, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans_input = input_tree.evaluate(variables)
                        ans_input_deriv = input_tree_derivative.evaluate(variables)
                        isclose(ans_input_deriv / cos(ans_input) ** 2, 1., abs_tol=1e-09)


class TestArcSine:
    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_evaluate(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = ArcSine(input_tree)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), asin(input_tree.evaluate(variables)), abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(asin(input_tree.evaluate(variables)), 1., abs_tol=1e-09)

    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_derivative(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = ArcSine(input_tree)
        for var in total_tree.dependencies():
            input_tree_derivative = input_tree.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_input = input_tree.evaluate(variables)
                    ans_input_deriv = input_tree_derivative.evaluate(variables)
                    assert isclose(ans_total, ans_input_deriv / (1 - ans_input ** 2) ** .5, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans_input = input_tree.evaluate(variables)
                        ans_input_deriv = input_tree_derivative.evaluate(variables)
                        isclose(ans_input_deriv / (1 - ans_input ** 2) ** .5, 1., abs_tol=1e-09)


class TestArcCosine:
    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_evaluate(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = ArcCosine(input_tree)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), acos(input_tree.evaluate(variables)), abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(acos(input_tree.evaluate(variables)), 1., abs_tol=1e-09)

    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_derivative(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = ArcCosine(input_tree)
        for var in total_tree.dependencies():
            input_tree_derivative = input_tree.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_input = input_tree.evaluate(variables)
                    ans_input_deriv = input_tree_derivative.evaluate(variables)
                    assert isclose(ans_total, -ans_input_deriv / (1 - ans_input ** 2) ** .5, abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans_input = input_tree.evaluate(variables)
                        ans_input_deriv = input_tree_derivative.evaluate(variables)
                        isclose(-ans_input_deriv / (1 - ans_input ** 2) ** .5, 1., abs_tol=1e-09)


class TestArcTangent:
    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_evaluate(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = ArcTangent(input_tree)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), atan(input_tree.evaluate(variables)), abs_tol=1e-09)
            except (ArithmeticError, ValueError):
                with pytest.raises(Exception):
                    isclose(atan(input_tree.evaluate(variables)), 1., abs_tol=1e-09)

    @pytest.mark.parametrize('rpn1', (x for x in rpn_list))
    def test_derivative(self, rpn1):
        input_tree = rpn_to_tree(rpn1)
        total_tree = ArcTangent(input_tree)
        for var in total_tree.dependencies():
            input_tree_derivative = input_tree.derivative(var)
            total_tree_derivative = total_tree.derivative(var)
            for variables in var_dicts_list:
                try:
                    ans_total = total_tree_derivative.evaluate(variables)
                    ans_input = input_tree.evaluate(variables)
                    ans_input_deriv = input_tree_derivative.evaluate(variables)
                    assert isclose(ans_total, ans_input_deriv / (ans_input ** 2 + 1), abs_tol=1e-09)
                except (ArithmeticError, ValueError):
                    with pytest.raises(Exception):
                        ans_input = input_tree.evaluate(variables)
                        ans_input_deriv = input_tree_derivative.evaluate(variables)
                        isclose(ans_input_deriv / (ans_input ** 2 + 1), 1., abs_tol=1e-09)

# todo: add substitution test
# todo: add infix test
# todo: add trig functions to tests

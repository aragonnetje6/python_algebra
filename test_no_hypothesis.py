import random
from math import isclose

import pytest

from interpreter import *

rpn_list = ['1 2 +',
            '2 3 * 4 5 * +',
            '5 x 2 ^ * 4 x * + 3 +',
            '5 5 x + y 4 / + - 2 * 5 ^ x *',
            '5 5 x + y 4 + + + 2 * 5 + x log 5 x 2 ^ * 4 x * + 3 + log']

precalc_tuple = [(1, 2, '+'),
                 ((2, 3, '*'), (4, 5, '*'), '+'),
                 (((5, ('x', 2, '^'), '*'), (4, 'x', '*'), '+'), 3, '+'),
                 ((((5, ((5, 'x', '+'), ('y', 4, '/'), '+'), '-'), 2, '*'), 5, '^'), 'x', '*'),
                 (((((5, ((5, 'x', '+'), ('y', 4, '+'), '+'), '+'), 2, '*'), 5, '+'), 'x', 'log'),
                  (((5, ('x', 2, '^'), '*'), (4, 'x', '*'), '+'), 3, '+'), 'log')]

precalc_ans = [3, 26, 276, -19736543.53125, 0.1383251008242367]

precalc_var_dict = {'x': 7,
                    'y': 11}

var_dicts_list = [{letter: random.randint(-100, 100) for letter in 'abcdefghijklmnopqrstuvwxyz'}]


class TestConversions:
    @pytest.mark.parametrize(('entry', 'output'), zip(rpn_list, precalc_tuple))
    def test_rpn_to_tupletree(self, entry, output):
        assert rpn_to_tuple(entry) == output

    @pytest.mark.parametrize(('entry', 'output'), zip(precalc_tuple, precalc_ans))
    def test_tuptree_to_ans(self, entry, output):
        assert tuple_to_ans(entry, precalc_var_dict) == output

    @pytest.mark.parametrize(('entry', 'output'), zip(rpn_list, precalc_ans))
    def test_rpn_to_ans(self, entry, output):
        assert rpn_to_ans(entry, precalc_var_dict) == output

    @pytest.mark.parametrize('entry', precalc_tuple)
    def test_tuptree_to_nodetree_reversible(self, entry):
        assert tuple_to_tree(entry).tuple() == entry

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_rpn_to_nodetree_reversible(self, rpn):
        assert rpn_to_tree(rpn).rpn() == rpn


class TestAddition:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Addition(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) + tree2.evaluate(variables))
            except (ArithmeticError, ValueError) as err:
                with pytest.raises(err.__class__):
                    isclose(tree1.evaluate(variables) + tree2.evaluate(variables), 1)

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
                    assert isclose(ans_total, ans_der1 + ans_der2)
                except (ArithmeticError, ValueError) as err:
                    with pytest.raises(err.__class__):
                        isclose(tree1_derivative.evaluate(variables) + tree2_derivative.evaluate(variables), 1)


class TestSubtraction:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Subtraction(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) - tree2.evaluate(variables))
            except (ArithmeticError, ValueError) as err:
                with pytest.raises(err.__class__):
                    isclose(tree1.evaluate(variables) - tree2.evaluate(variables), 1)

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
                    assert isclose(ans_total, ans_der1 - ans_der2)
                except (ArithmeticError, ValueError) as err:
                    with pytest.raises(err.__class__):
                        isclose(tree1_derivative.evaluate(variables) - tree2_derivative.evaluate(variables), 1)


class TestProduct:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Product(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) * tree2.evaluate(variables))
            except (ArithmeticError, ValueError) as err:
                with pytest.raises(err.__class__):
                    isclose(tree1.evaluate(variables) * tree2.evaluate(variables), 1)

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
                    assert isclose(ans_total, ans_der1 * ans2 + ans1 * ans_der2)
                except (ArithmeticError, ValueError) as err:
                    with pytest.raises(err.__class__):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        isclose(ans_der1 * ans2 + ans1 * ans_der2, 1)


class TestDivision:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Division(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) / tree2.evaluate(variables))
            except (ArithmeticError, ValueError) as err:
                with pytest.raises(err.__class__):
                    isclose(tree1.evaluate(variables) / tree2.evaluate(variables), 1)

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
                    assert isclose(ans_total, (ans_der1 * ans2 + ans1 * ans_der2) / ans2 ** 2)
                except (ArithmeticError, ValueError) as err:
                    with pytest.raises(err.__class__):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        isclose((ans_der1 * ans2 + ans1 * ans_der2) / ans2 ** 2, 1)


class TestExponent:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Exponent(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) ** tree2.evaluate(variables))
            except (ArithmeticError, ValueError) as err:
                with pytest.raises(err.__class__):
                    isclose(tree1.evaluate(variables) ** tree2.evaluate(variables), 1)

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
                    assert isclose(ans_total, ans1 ** (ans2 - 1) * (ans2 * ans_der1 + ans1 * log(ans1) * ans_der2))
                except (ArithmeticError, ValueError) as err:
                    with pytest.raises(err.__class__):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        isclose(ans1 ** (ans2 - 1) * (ans2 * ans_der1 + ans1 * log(ans1) * ans_der2), 1)


class TestLogarithm:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Logarithm(tree1, tree2)
        for variables in var_dicts_list:
            try:
                assert isclose(total_tree.evaluate(variables),
                               log(tree1.evaluate(variables), tree2.evaluate(variables)))
            except (ArithmeticError, ValueError) as err:
                with pytest.raises(err.__class__):
                    isclose(log(tree1.evaluate(variables), tree2.evaluate(variables)), 1)

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
                                   ((ans_der1 * log(ans2) / ans1) - (ans_der2 * log(ans1) / ans2)) / log(ans2) ** 2)
                except (ArithmeticError, ValueError) as err:
                    with pytest.raises(err.__class__):
                        ans1 = tree1.evaluate(variables)
                        ans2 = tree2.evaluate(variables)
                        ans_der1 = tree1_derivative.evaluate(variables)
                        ans_der2 = tree2_derivative.evaluate(variables)
                        isclose(((ans_der1 * log(ans2) / ans1) - (ans_der2 * log(ans1) / ans2)) / log(ans2) ** 2, 1)


# todo: add trig tests

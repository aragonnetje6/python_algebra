import random
from math import isclose

import pytest

from interpreter import *


def random_expression(depth=0):
    if random.randint(0, 2) == 0:
        return random_term(depth), random_term(depth), random.choice(list(operator_2_in_classes.keys()))
    else:
        return random_term(depth), random.choice(list(operator_1_in_classes.keys()))


def random_term(depth=0):
    choice = random.randint(0, 8 - depth)
    if choice == 0:
        return random.randint(-100, 101)
    elif choice == 1:
        return random.random() * 200 - 100
    elif choice == 2:
        return random.choice('abcdefghijklmnopqrstuvwxyz')
    else:
        return random_expression(depth + 1)


def tuple_to_rpn(tup):
    out = ''
    for item in tup:
        out += ' '
        if isinstance(item, str):
            out += item
        elif isinstance(item, (int, float)):
            out += str(item)
        elif isinstance(item, tuple):
            out += tuple_to_rpn(item)
    return out.strip()


def check_overflow(tup):
    tree = tuple_to_tree(tup)
    try:
        for variables in var_dicts_list:
            float(tree.evaluate(variables))
    except OverflowError:
        return False
    except:
        pass
    return True


precalc_rpn = ['1 2 +',
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

var_dicts_list = [{letter: random.random() * 200 - 100 for letter in 'abcdefghijklmnopqrstuvwxyz'} for i in range(20)]
tuples_list = list(filter(check_overflow, [random_expression() for j in range(10)]))
unsafe_rpn_list = [tuple_to_rpn(tup) for tup in tuples_list]
rpn_list = list(
    filter(lambda x: all(rpn_to_tree(x).validate(variables) for variables in var_dicts_list), unsafe_rpn_list))

while len(rpn_list) < 10:
    item = random_expression()
    if check_overflow(item) and all(tuple_to_tree(item).validate(variables) for variables in var_dicts_list):
        rpn_list.append(tuple_to_rpn(item))


class TestConversions:
    @pytest.mark.parametrize(('entry', 'output'), zip(precalc_rpn, precalc_tuple))
    def test_rpn_to_tupletree(self, entry, output):
        assert rpn_to_tuple(entry) == output

    @pytest.mark.parametrize(('entry', 'output'), zip(precalc_tuple, precalc_ans))
    def test_tuptree_to_ans(self, entry, output):
        assert tuple_to_ans(entry, precalc_var_dict) == output

    @pytest.mark.parametrize(('entry', 'output'), zip(precalc_rpn, precalc_ans))
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
    def test_validate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Addition(tree1, tree2)
        for variables in var_dicts_list:
            assert total_tree.validate(variables) == tree1.validate(variables) or \
                   total_tree.validate(variables) == tree2.validate(variables)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Addition(tree1, tree2)
        for variables in var_dicts_list:
            assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) + tree2.evaluate(variables))

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
                ans_der1 = tree1_derivative.evaluate(variables)
                ans_der2 = tree2_derivative.evaluate(variables)
                assert isclose(total_tree_derivative.evaluate(variables),
                               ans_der1 + ans_der2)


class TestSubtraction:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_validate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Subtraction(tree1, tree2)
        for variables in var_dicts_list:
            assert total_tree.validate(variables) == tree1.validate(variables) or \
                   total_tree.validate(variables) == tree2.validate(variables)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Subtraction(tree1, tree2)
        for variables in var_dicts_list:
            assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) - tree2.evaluate(variables))

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
                ans_der1 = tree1_derivative.evaluate(variables)
                ans_der2 = tree2_derivative.evaluate(variables)
                assert isclose(total_tree_derivative.evaluate(variables),
                               ans_der1 - ans_der2)


class TestProduct:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_validate(self, rpn1, rpn2):
        # todo: add validation test
        pass

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Product(tree1, tree2)
        for variables in var_dicts_list:
            assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) * tree2.evaluate(variables))

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
                ans1 = tree1.evaluate(variables)
                ans2 = tree2.evaluate(variables)
                ans_der1 = tree1_derivative.evaluate(variables)
                ans_der2 = tree2_derivative.evaluate(variables)
                assert isclose(total_tree_derivative.evaluate(variables),
                               ans_der1 * ans2 + ans1 * ans_der2)


class TestDivision:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_validate(self, rpn1, rpn2):
        # todo: add validation test
        pass

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Product(tree1, tree2)
        for variables in var_dicts_list:
            if total_tree.validate(variables):
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) / tree2.evaluate(variables))

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
                if total_tree_derivative.validate(variables):
                    ans1 = tree1.evaluate(variables)
                    ans2 = tree2.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(total_tree_derivative.evaluate(variables),
                                   (ans_der1 * ans2 - ans1 * ans_der2) / ans2 ** 2)


class TestExponent:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_validate(self, rpn1, rpn2):
        # todo: add validation test
        pass

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Product(tree1, tree2)
        for variables in var_dicts_list:
            if total_tree.validate(variables):
                assert isclose(total_tree.evaluate(variables), tree1.evaluate(variables) ** tree2.evaluate(variables))

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
                if total_tree_derivative.validate(variables):
                    ans1 = tree1.evaluate(variables)
                    ans2 = tree2.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(total_tree_derivative.evaluate(variables),
                                   ans1 ** (ans2 - 1) * (ans2 * ans_der1 + ans1 * log(ans1) * ans_der2))


class TestLogarithm:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_validate(self, rpn1, rpn2):
        # todo: add validation test
        pass

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_evaluate(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        total_tree = Product(tree1, tree2)
        for variables in var_dicts_list:
            if total_tree.validate(variables):
                assert isclose(total_tree.evaluate(variables),
                               log(tree1.evaluate(variables), tree2.evaluate(variables)))

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
                if total_tree_derivative.validate(variables):
                    ans1 = tree1.evaluate(variables)
                    ans2 = tree2.evaluate(variables)
                    ans_der1 = tree1_derivative.evaluate(variables)
                    ans_der2 = tree2_derivative.evaluate(variables)
                    assert isclose(total_tree_derivative.evaluate(variables),
                                   ((ans_der1 * log(ans2) / ans1) - (ans_der2 * log(ans1) / ans2))) / log(ans2) ** 2


@pytest.mark.parametrize('rpn', rpn_list)
def test_total_derivative(rpn):
    tree = rpn_to_tree(rpn)
    total_derivative = tree.total_derivative()
    for variables in var_dicts_list:
        if tree.validate(variables) and total_derivative.validate(variables):
            result = total_derivative.evaluate(variables)
            reference = sum(tree.derivative(var).evaluate(variables) for var in tree.dependencies())
            assert result == reference

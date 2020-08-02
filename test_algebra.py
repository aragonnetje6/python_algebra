import random
from hypothesis import strategies as st
from math import isclose

import pytest

from python_algebra.v5.interpreter import *


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

expression = st.builds(tuple_to_rpn, st.recursive(
    st.one_of(st.integers(), st.floats(), st.characters(min_codepoint=97, max_codepoint=122)),
    lambda exp: st.one_of(
        st.tuples(exp, exp, st.sampled_from(list(operator_2_in_classes.keys()))),
        st.tuples(exp, st.sampled_from(list(operator_1_in_classes.keys())))))).filter(check_overflow).filter(
    lambda x: all(rpn_to_tree(x).validate(variables) for variables in var_dicts_list))


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
    def test_commutative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        sum1 = Addition(tree1, tree2)
        sum2 = Addition(tree2, tree1)
        for variables in var_dicts_list:
            assert isclose(sum1.evaluate(variables), sum2.evaluate(variables))

    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_associative(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        sum1 = Addition(tree1, Addition(tree2, tree3))
        sum2 = Addition(tree3, Addition(tree1, tree2))
        sum3 = Addition(tree2, Addition(tree1, tree3))
        for variables in var_dicts_list:
            assert isclose(sum1.evaluate(variables), sum2.evaluate(variables)) and isclose(sum1.evaluate(variables),
                                                                                           sum3.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_identity(self, rpn):
        tree = rpn_to_tree(rpn)
        identity = Constant(0)
        sum1 = Addition(tree, identity)
        for variables in var_dicts_list:
            assert isclose(sum1.evaluate(variables), tree.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_successor(self, rpn):
        tree = rpn_to_tree(rpn)
        sum1 = Addition(Addition(tree, Constant(1)), Constant(1))
        sum2 = Addition(tree, Constant(2))
        for variables in var_dicts_list:
            assert isclose(sum1.evaluate(variables), sum2.evaluate(variables))


class TestSubtraction:
    @pytest.mark.parametrize('rpn', rpn_list)
    def test_identity(self, rpn):
        tree = rpn_to_tree(rpn)
        identity = Constant(0)
        subtraction1 = Subtraction(tree, identity)
        for variables in var_dicts_list:
            assert isclose(subtraction1.evaluate(variables), tree.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_zero(self, rpn):
        tree = rpn_to_tree(rpn)
        subtraction1 = Subtraction(tree, tree)
        for variables in var_dicts_list:
            assert subtraction1.evaluate(variables) == 0

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_successor(self, rpn):
        tree = rpn_to_tree(rpn)
        subtraction1 = Subtraction(Subtraction(tree, Constant(1)), Constant(1))
        subtraction2 = Subtraction(tree, Constant(2))
        for variables in var_dicts_list:
            assert isclose(subtraction1.evaluate(variables), subtraction2.evaluate(variables))

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_inverse_addition(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        subtraction1 = Subtraction(Addition(tree1, tree2), tree2)
        for variables in var_dicts_list:
            if subtraction1.validate(variables):
                assert isclose(subtraction1.evaluate(variables), tree1.evaluate(variables))


class TestProduct:
    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_commutative(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        product1 = Product(tree1, tree2)
        product2 = Product(tree2, tree1)
        for variables in var_dicts_list:
            assert isclose(product1.evaluate(variables), product2.evaluate(variables))

    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_associative(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        product1 = Product(tree1, Product(tree2, tree3))
        product2 = Product(tree3, Product(tree1, tree2))
        product3 = Product(tree2, Product(tree1, tree3))
        for variables in var_dicts_list:
            assert isclose(product1.evaluate(variables), product2.evaluate(variables)) and \
                   isclose(product1.evaluate(variables), product3.evaluate(variables))

    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_distributive(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        product1 = Product(tree1, Addition(tree2, tree3))
        product2 = Addition(Product(tree1, tree2), Product(tree1, tree3))
        for variables in var_dicts_list:
            assert isclose(product1.evaluate(variables), product2.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_identity(self, rpn):
        tree = rpn_to_tree(rpn)
        identity = Constant(1)
        product1 = Product(tree, identity)
        for variables in var_dicts_list:
            assert isclose(product1.evaluate(variables), tree.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_zero(self, rpn):
        tree = rpn_to_tree(rpn)
        zero = Constant(0)
        product1 = Product(tree, zero)
        for variables in var_dicts_list:
            assert product1.evaluate(variables) == 0

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_negation(self, rpn):
        tree = rpn_to_tree(rpn)
        minus_one = Constant(-1)
        product1 = Product(tree, minus_one)
        for variables in var_dicts_list:
            assert isclose(product1.evaluate(variables), -tree.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_doubling(self, rpn):
        tree = rpn_to_tree(rpn)
        two = Constant(2)
        product1 = Product(tree, two)
        sum1 = Addition(tree, tree)
        for variables in var_dicts_list:
            assert isclose(product1.evaluate(variables), sum1.evaluate(variables))

    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_order(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        product1 = Product(tree1, tree2)
        product2 = Product(tree1, tree3)
        for variables in var_dicts_list:
            if tree1.evaluate(variables) > 0:
                if tree2.evaluate(variables) > tree3.evaluate(variables):
                    assert product1.evaluate(variables) > product2.evaluate(variables)
                elif tree2.evaluate(variables) == tree3.evaluate(variables):
                    assert product1.evaluate(variables) == product2.evaluate(variables)
                elif tree2.evaluate(variables) < tree3.evaluate(variables):
                    assert product1.evaluate(variables) < product2.evaluate(variables)
            elif tree1.evaluate(variables) < 0:
                if tree2.evaluate(variables) > tree3.evaluate(variables):
                    assert product1.evaluate(variables) < product2.evaluate(variables)
                elif tree2.evaluate(variables) == tree3.evaluate(variables):
                    assert isclose(product1.evaluate(variables), product2.evaluate(variables))
                elif tree2.evaluate(variables) < tree3.evaluate(variables):
                    assert product1.evaluate(variables) > product2.evaluate(variables)


class TestDivision:
    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_distributive(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        division1 = Division(Addition(tree2, tree3), tree1)
        division2 = Addition(Division(tree2, tree1), Division(tree3, tree1))
        for variables in var_dicts_list:
            if division1.validate(variables) and division2.validate(variables):
                assert isclose(division1.evaluate(variables), division2.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_identity(self, rpn):
        tree = rpn_to_tree(rpn)
        identity = Constant(1)
        division1 = Division(tree, identity)
        for variables in var_dicts_list:
            if division1.validate(variables):
                assert isclose(division1.evaluate(variables), tree.evaluate(variables))

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_inverse_multiplication(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        division1 = Division(Product(tree1, tree2), tree2)
        for variables in var_dicts_list:
            if division1.validate(variables):
                assert isclose(division1.evaluate(variables), tree1.evaluate(variables))


class TestExponent:
    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_distributive(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exponent1 = Exponent(Product(tree2, tree3), tree1)
        exponent2 = Product(Exponent(tree2, tree1), Exponent(tree3, tree1))
        for variables in var_dicts_list:
            try:
                if exponent1.validate(variables) and exponent2.validate(variables):
                    assert isclose(exponent1.evaluate(variables), exponent2.evaluate(variables), abs_tol=1e-10)
            except OverflowError:
                pass

    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_additive(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exponent1 = Product(Exponent(tree1, tree2), Exponent(tree1, tree3))
        exponent2 = Exponent(tree1, Addition(tree2, tree3))
        for variables in var_dicts_list:
            try:
                if exponent1.validate(variables) and exponent2.validate(variables):
                    assert isclose(exponent1.evaluate(variables), exponent2.evaluate(variables), abs_tol=1e-10)
            except OverflowError:
                pass

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_identity(self, rpn):
        tree = rpn_to_tree(rpn)
        identity = Constant(1)
        exponent1 = Exponent(tree, identity)
        for variables in var_dicts_list:
            try:
                if exponent1.validate(variables):
                    assert isclose(exponent1.evaluate(variables), tree.evaluate(variables))
            except OverflowError:
                pass

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_one(self, rpn):
        tree = rpn_to_tree(rpn)
        identity = Constant(1)
        exponent1 = Exponent(identity, tree)
        for variables in var_dicts_list:
            try:
                if exponent1.validate(variables):
                    assert exponent1.evaluate(variables) == 1
            except OverflowError:
                pass

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_zero(self, rpn):
        tree = rpn_to_tree(rpn)
        zero = Constant(0)
        exponent1 = Exponent(tree, zero)
        for variables in var_dicts_list:
            try:
                if exponent1.validate(variables):
                    assert exponent1.evaluate(variables) == 1
            except OverflowError:
                pass

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_zero_2(self, rpn):
        tree = rpn_to_tree(rpn)
        zero = Constant(0)
        exponent1 = Exponent(zero, tree)
        for variables in var_dicts_list:
            try:
                if exponent1.validate(variables):
                    assert exponent1.evaluate(variables) == 0
            except OverflowError:
                pass

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_inversion(self, rpn):
        tree = rpn_to_tree(rpn)
        negative = Constant(-1)
        exponent1 = Exponent(tree, negative)
        for variables in var_dicts_list:
            try:
                if exponent1.validate(variables):
                    assert isclose(exponent1.evaluate(variables), 1 / tree.evaluate(variables))
            except OverflowError:
                pass


class TestLogarithm:
    @pytest.mark.parametrize(('rpn1', 'rpn2', 'rpn3'),
                             ((x, y, z) for x in rpn_list for y in rpn_list for z in rpn_list))
    def test_distributive(self, rpn1, rpn2, rpn3):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        logarithm1 = Logarithm(Product(tree2, tree3), tree1)
        logarithm2 = Addition(Logarithm(tree2, tree1), Logarithm(tree3, tree1))
        for variables in var_dicts_list:
            if logarithm1.validate(variables) and logarithm2.validate(variables):
                assert isclose(logarithm1.evaluate(variables), logarithm2.evaluate(variables))

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_fraction(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        logarithm1 = Logarithm(tree1, tree2)
        logarithm2 = Division(Logarithm(tree1, Constant(e)), Logarithm(tree2, Constant(e)))
        for variables in var_dicts_list:
            if logarithm1.validate(variables) and logarithm2.validate(variables):
                assert isclose(logarithm1.evaluate(variables), logarithm2.evaluate(variables))

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_one(self, rpn):
        tree = rpn_to_tree(rpn)
        logarithm1 = Logarithm(tree, tree)
        for variables in var_dicts_list:
            if logarithm1.validate(variables):
                assert logarithm1.evaluate(variables) == 1

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_one(self, rpn):
        tree = rpn_to_tree(rpn)
        one = Constant(1)
        logarithm1 = Logarithm(one, tree)
        for variables in var_dicts_list:
            if logarithm1.validate(variables):
                assert logarithm1.evaluate(variables) == 0

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_inverse_exponent(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        logarithm1 = Logarithm(Exponent(tree1, tree2), tree1)
        for variables in var_dicts_list:
            try:
                if logarithm1.validate(variables):
                    assert isclose(logarithm1.evaluate(variables), tree2.evaluate(variables), abs_tol=1e-5)
            except OverflowError:
                pass


class TestValidation:
    @pytest.mark.parametrize('rpn', unsafe_rpn_list)
    def test_validation_general(self, rpn):
        tree = rpn_to_tree(rpn)
        for variables in var_dicts_list:
            if tree.validate(variables):
                tree.evaluate(variables)
            else:
                with pytest.raises(Exception):
                    assert tree.evaluate(variables)

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in unsafe_rpn_list for y in unsafe_rpn_list))
    def test_validate_product(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        prodtree = Product(tree1, tree2)

        for variables in var_dicts_list:
            tree1_valid = tree1.validate(variables)
            tree2_valid = tree2.validate(variables)
            prodtree_valid = prodtree.validate(variables)

            if prodtree_valid:
                if not tree1_valid:
                    assert tree2.evaluate(variables) == 0
                if not tree2_valid:
                    assert tree1.evaluate(variables) == 0
            else:
                if tree1_valid:
                    assert not tree2_valid and tree1.evaluate(variables) != 0
                if tree2_valid:
                    assert not tree1_valid and tree2.evaluate(variables) != 0

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in unsafe_rpn_list for y in unsafe_rpn_list))
    def test_validate_division(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        divtree = Division(tree1, tree2)

        for variables in var_dicts_list:
            tree1_valid = tree1.validate(variables)
            tree2_valid = tree2.validate(variables)
            divtree_valid = divtree.validate(variables)
            if tree1_valid and tree2_valid:
                assert divtree_valid == (tree2.evaluate(variables) != 0)
            else:
                assert not divtree_valid

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in unsafe_rpn_list for y in unsafe_rpn_list))
    def test_validate_exponent(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exptree = Exponent(tree1, tree2)

        try:
            for variables in var_dicts_list:
                tree1_valid = tree1.validate(variables)
                tree2_valid = tree2.validate(variables)
                exptree_valid = exptree.validate(variables)
                if exptree_valid:
                    if not tree1_valid:
                        assert tree2.evaluate(variables) == 0
                    elif not tree2_valid:
                        tree1_result = tree1.evaluate(variables)
                        assert tree1_result == 0 or tree1_result == 1
                    else:
                        assert tree1.evaluate(variables) >= 0 or tree2.evaluate(variables) % 1 == 0
                else:
                    if tree1_valid:
                        tree1_result = tree1.evaluate(variables)
                        if tree2_valid:
                            assert tree2.evaluate(variables) % 1 != 0 and tree1_result < 0
                        else:
                            assert not tree2_valid and tree1_result != 0 and tree1_result != 1
                    elif tree2_valid:
                        assert not tree1_valid and tree2.evaluate(variables) != 0
        except OverflowError:
            pass

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in unsafe_rpn_list for y in unsafe_rpn_list))
    def test_validate_log(self, rpn1, rpn2):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        logtree = Logarithm(tree1, tree2)

        for variables in var_dicts_list:
            tree1_valid = tree1.validate(variables)
            tree2_valid = tree2.validate(variables)
            logtree_valid = logtree.validate(variables)
            if tree1_valid and tree2_valid:
                tree1_result = tree1.evaluate(variables)
                tree2_result = tree2.evaluate(variables)
                assert logtree_valid == (tree1_result > 0 and tree2_result > 0 and tree2_result != 1)
            else:
                assert not logtree_valid

    @pytest.mark.parametrize('rpn', unsafe_rpn_list)
    def test_validate_tan(self, rpn):
        tree1 = rpn_to_tree(rpn)
        tantree = Tangent(tree1)
        for variables in var_dicts_list:
            if tree1.validate(variables):
                assert tantree.validate(variables) == (tree1.evaluate(variables) % pi != pi / 2)
            else:
                assert not tantree.validate(variables)

    @pytest.mark.parametrize('rpn', unsafe_rpn_list)
    def test_validate_asin(self, rpn):
        tree1 = rpn_to_tree(rpn)
        asintree = ArcSine(tree1)
        for variables in var_dicts_list:
            if tree1.validate(variables):
                assert asintree.validate(variables) == (-1 <= tree1.evaluate(variables) <= 1)
            else:
                assert not asintree.validate(variables)

    @pytest.mark.parametrize('rpn', unsafe_rpn_list)
    def test_validate_acos(self, rpn):
        tree1 = rpn_to_tree(rpn)
        acostree = ArcCosine(tree1)
        for variables in var_dicts_list:
            if tree1.validate(variables):
                assert acostree.validate(variables) == (-1 <= tree1.evaluate(variables) <= 1)
            else:
                assert not acostree.validate(variables)


class TestDerivatives:
    @pytest.mark.parametrize('rpn', rpn_list)
    def test_derivative_constant_rule(self, rpn):
        f = rpn_to_tree(rpn)
        for variables in var_dicts_list:
            if f.validate(variables):
                result = f.derivative('').evaluate(variables)
                assert result == 0

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative_sum_rule(self, rpn1, rpn2):
        f = rpn_to_tree(rpn1)
        g = rpn_to_tree(rpn2)
        sum_f_g = Addition(f, g)
        for variables in var_dicts_list:
            if sum_f_g.validate(variables):
                for var in f.dependencies().union(g.dependencies()):
                    derivative = sum_f_g.derivative(var)
                    if derivative.validate(variables):
                        result = sum_f_g.derivative(var).evaluate(variables)
                        reference = f.derivative(var).evaluate(variables) + g.derivative(var).evaluate(variables)
                        assert result == reference

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative_product_rule(self, rpn1, rpn2):
        f = rpn_to_tree(rpn1)
        g = rpn_to_tree(rpn2)
        product_f_g = Product(f, g)
        for variables in var_dicts_list:
            if product_f_g.validate(variables) and f.validate(variables) and g.validate(variables):
                for var in product_f_g.dependencies():
                    derivative = product_f_g.derivative(var)
                    if derivative.validate(variables):
                        result = product_f_g.derivative(var).evaluate(variables)
                        reference = f.derivative(var).evaluate(variables) * g.evaluate(variables) \
                                    + f.evaluate(variables) * g.derivative(var).evaluate(variables)
                        assert result == reference

    @pytest.mark.parametrize(('rpn1', 'rpn2'), ((x, y) for x in rpn_list for y in rpn_list))
    def test_derivative_quotient_rule(self, rpn1, rpn2):
        f = rpn_to_tree(rpn1)
        g = rpn_to_tree(rpn2)
        quotient_f_g = Division(f, g)
        for variables in var_dicts_list:
            if quotient_f_g.validate(variables):
                for var in f.dependencies().union(g.dependencies()):
                    derivative = quotient_f_g.derivative(var)
                    if derivative.validate(variables):
                        result = derivative.evaluate(variables)
                        reference = (f.derivative(var).evaluate(variables) * g.evaluate(variables)
                                     - f.evaluate(variables) * g.derivative(var).evaluate(variables)) \
                                    / g.evaluate(variables) ** 2
                        assert result == reference

    @pytest.mark.parametrize(('rpn', 'operator'), ((x, y) for x in rpn_list for y in operator_1_in_classes.values()))
    def test_derivative_chain_rule(self, rpn, operator):
        g = rpn_to_tree(rpn)
        h = operator(g)
        for variables in var_dicts_list:
            if h.validate(variables):
                for var in h.dependencies():
                    derivative = h.derivative(var)
                    if derivative.validate(variables):
                        result = derivative.evaluate(variables)
                        g_result = g.evaluate(variables)
                        f = operator(Variable(var))
                        reference = f.derivative(var).evaluate({**variables, var: g_result}) * g.derivative(
                            var).evaluate(
                            variables)
                        assert isclose(result, reference)

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_total_derivative(self, rpn):
        tree = rpn_to_tree(rpn)
        total_derivative = tree.total_derivative()
        for variables in var_dicts_list:
            if tree.validate(variables) and total_derivative.validate(variables):
                result = total_derivative.evaluate(variables)
                reference = sum(tree.derivative(var).evaluate(variables) for var in tree.dependencies())
                assert result == reference

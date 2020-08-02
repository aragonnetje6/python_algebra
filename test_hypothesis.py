from hypothesis import given
from hypothesis import strategies as st
from math import isclose

import pytest

from python_algebra.v5.interpreter import *

example_dict = {letter: st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)) for letter in 'abcdefghijklmnopqrstuvwxyz'}
var_dict = st.fixed_dictionaries(example_dict)

term = st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.characters(min_codepoint=97, max_codepoint=122))
tuple_expression = st.recursive(term, lambda exp: st.one_of(
    st.tuples(exp, exp, st.sampled_from(list(operator_2_in_classes.keys()))),
    st.tuples(exp, st.sampled_from(list(operator_1_in_classes.keys())))), max_leaves=8).filter(
    lambda x: type(x) == tuple)
rpn_expression = st.builds(tuple_to_rpn, tuple_expression).filter(lambda x: rpn_to_tree(x).validate({letter: 1  for letter in 'abcdefghijklmnopqrstuvwxyz'}))


class TestConversions:
    @given(rpn_expression)
    def test_rpn_to_nodetree_reversible(self, rpn):
        assert rpn_to_tree(rpn).rpn() == rpn


class TestAddition:
    @given(rpn_expression, rpn_expression, var_dict)
    def test_commutative(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exp1 = Addition(tree1, tree2)
        exp2 = Addition(tree2, tree1)
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))

    @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
    def test_associative(self, rpn1, rpn2, rpn3, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exp1 = Addition(tree1, Addition(tree2, tree3))
        exp2 = Addition(tree3, Addition(tree1, tree2))
        exp3 = Addition(tree2, Addition(tree1, tree3))
        if exp1.validate(variables) and exp2.validate(variables) and exp3.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables)) and \
                   isclose(exp1.evaluate(variables), exp3.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_identity(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        identity = Constant(0)
        exp1 = Addition(tree, identity)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), tree.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_successor(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        exp1 = Addition(Addition(tree, Constant(1)), Constant(1))
        exp2 = Addition(tree, Constant(2))
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))


class TestSubtraction:
    @given(rpn_expression, var_dict)
    def test_identity(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        identity = Constant(0)
        exp1 = Subtraction(tree, identity)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), tree.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_zero(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        exp1 = Subtraction(tree, tree)
        if exp1.validate(variables):
            assert exp1.evaluate(variables) == 0

    @given(rpn_expression, var_dict)
    def test_successor(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        exp1 = Subtraction(Subtraction(tree, Constant(1)), Constant(1))
        exp2 = Subtraction(tree, Constant(2))
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))

    @given(rpn_expression, rpn_expression, var_dict)
    def test_inverse_addition(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exp1 = Subtraction(Addition(tree1, tree2), tree2)
        exp2 = Addition(Subtraction(tree1, tree2), tree2)
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), tree1.evaluate(variables))
            assert isclose(exp2.evaluate(variables), tree1.evaluate(variables))


class TestProduct:
    @given(rpn_expression, rpn_expression, var_dict)
    def test_commutative(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exp1 = Product(tree1, tree2)
        exp2 = Product(tree2, tree1)
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))

    @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
    def test_associative(self, rpn1, rpn2, rpn3, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exp1 = Product(tree1, Product(tree2, tree3))
        exp2 = Product(tree3, Product(tree1, tree2))
        exp3 = Product(tree2, Product(tree1, tree3))
        if exp1.validate(variables) and exp2.validate(variables) and exp3.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables)) and \
                   isclose(exp1.evaluate(variables), exp3.evaluate(variables))

    @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
    def test_distributive(self, rpn1, rpn2, rpn3, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exp1 = Product(tree1, Addition(tree2, tree3))
        exp2 = Addition(Product(tree1, tree2), Product(tree1, tree3))
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_identity(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        identity = Constant(1)
        exp1 = Product(tree, identity)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), tree.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_zero(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        zero = Constant(0)
        exp1 = Product(tree, zero)
        if exp1.validate(variables):
            assert exp1.evaluate(variables) == 0

    @given(rpn_expression, var_dict)
    def test_negation(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        minus_one = Constant(-1)
        exp1 = Product(tree, minus_one)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), -tree.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_doubling(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        two = Constant(2)
        exp1 = Product(tree, two)
        sum1 = Addition(tree, tree)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), sum1.evaluate(variables))

    @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
    def test_order(self, rpn1, rpn2, rpn3, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exp1 = Product(tree1, tree2)
        exp2 = Product(tree1, tree3)
        if exp1.validate(variables) and exp2.validate(variables):
            if tree1.evaluate(variables) > 0:
                if tree2.evaluate(variables) > tree3.evaluate(variables):
                    assert exp1.evaluate(variables) > exp2.evaluate(variables)
                elif tree2.evaluate(variables) == tree3.evaluate(variables):
                    assert exp1.evaluate(variables) == exp2.evaluate(variables)
                elif tree2.evaluate(variables) < tree3.evaluate(variables):
                    assert exp1.evaluate(variables) < exp2.evaluate(variables)
            elif tree1.evaluate(variables) < 0:
                if tree2.evaluate(variables) > tree3.evaluate(variables):
                    assert exp1.evaluate(variables) < exp2.evaluate(variables)
                elif tree2.evaluate(variables) == tree3.evaluate(variables):
                    assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))
                elif tree2.evaluate(variables) < tree3.evaluate(variables):
                    assert exp1.evaluate(variables) > exp2.evaluate(variables)


class TestDivision:
    @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
    def test_distributive(self, rpn1, rpn2, rpn3, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exp1 = Division(Addition(tree2, tree3), tree1)
        exp2 = Addition(Division(tree2, tree1), Division(tree3, tree1))
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_identity(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        identity = Constant(1)
        exp1 = Division(tree, identity)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), tree.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_identity_2(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        identity = Constant(1)
        exp1 = Division(identity, tree)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), tree.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_zero(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        zero = Constant(0)
        exp1 = Division(zero, tree)
        if exp1.validate(variables):
            assert exp1.evaluate(variables) == 0

    @given(rpn_expression, rpn_expression, var_dict)
    def test_inverse_multiplication(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exp1 = Division(Product(tree1, tree2), tree2)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), tree1.evaluate(variables))


# class TestExponent:
#     @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
#     def test_distributive(self, rpn1, rpn2, rpn3, variables):
#         tree1 = rpn_to_tree(rpn1)
#         tree2 = rpn_to_tree(rpn2)
#         tree3 = rpn_to_tree(rpn3)
#         exp1 = Exponent(Product(tree2, tree3), tree1)
#         exp2 = Product(Exponent(tree2, tree1), Exponent(tree3, tree1))
#         if exp1.validate(variables) and exp2.validate(variables):
#             assert isclose(exp1.evaluate(variables), exp2.evaluate(variables), abs_tol=1e-10)
#
#     @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
#     def test_additive(self, rpn1, rpn2, rpn3, variables):
#         tree1 = rpn_to_tree(rpn1)
#         tree2 = rpn_to_tree(rpn2)
#         tree3 = rpn_to_tree(rpn3)
#         exp1 = Product(Exponent(tree1, tree2), Exponent(tree1, tree3))
#         exp2 = Exponent(tree1, Addition(tree2, tree3))
#         if exp1.validate(variables) and exp2.validate(variables):
#             assert isclose(exp1.evaluate(variables), exp2.evaluate(variables), abs_tol=1e-10)
#
#     @given(rpn_expression, var_dict)
#     def test_identity(self, rpn, variables):
#         tree = rpn_to_tree(rpn)
#         identity = Constant(1)
#         exp1 = Exponent(tree, identity)
#         if exp1.validate(variables):
#             assert isclose(exp1.evaluate(variables), tree.evaluate(variables))
#
#     @given(rpn_expression, var_dict)
#     def test_one(self, rpn, variables):
#         tree = rpn_to_tree(rpn)
#         identity = Constant(1)
#         exp1 = Exponent(identity, tree)
#         if exp1.validate(variables):
#             assert exp1.evaluate(variables) == 1
#
#     @given(rpn_expression, var_dict)
#     def test_zero(self, rpn, variables):
#         tree = rpn_to_tree(rpn)
#         zero = Constant(0)
#         exp1 = Exponent(tree, zero)
#         if exp1.validate(variables):
#             assert exp1.evaluate(variables) == 1
#
#     @given(rpn_expression, var_dict)
#     def test_zero_2(self, rpn, variables):
#         tree = rpn_to_tree(rpn)
#         zero = Constant(0)
#         exp1 = Exponent(zero, tree)
#         if exp1.validate(variables):
#             assert exp1.evaluate(variables) == 0
#
#     @given(rpn_expression, var_dict)
#     def test_inversion(self, rpn, variables):
#         tree = rpn_to_tree(rpn)
#         negative = Constant(-1)
#         exp1 = Exponent(tree, negative)
#         if exp1.validate(variables):
#             assert isclose(exp1.evaluate(variables), 1 / tree.evaluate(variables))


class TestLogarithm:
    @given(rpn_expression, rpn_expression, rpn_expression, var_dict)
    def test_distributive(self, rpn1, rpn2, rpn3, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        tree3 = rpn_to_tree(rpn3)
        exp1 = Logarithm(Product(tree2, tree3), tree1)
        exp2 = Addition(Logarithm(tree2, tree1), Logarithm(tree3, tree1))
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))

    @given(rpn_expression, rpn_expression, var_dict)
    def test_fraction(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exp1 = Logarithm(tree1, tree2)
        exp2 = Division(Logarithm(tree1, Constant(e)), Logarithm(tree2, Constant(e)))
        if exp1.validate(variables) and exp2.validate(variables):
            assert isclose(exp1.evaluate(variables), exp2.evaluate(variables))

    @given(rpn_expression, var_dict)
    def test_one(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        exp1 = Logarithm(tree, tree)
        if exp1.validate(variables):
            assert exp1.evaluate(variables) == 1

    @given(rpn_expression, var_dict)
    def test_one(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        one = Constant(1)
        exp1 = Logarithm(one, tree)
        if exp1.validate(variables):
            assert exp1.evaluate(variables) == 0

    @given(rpn_expression, rpn_expression, var_dict)
    def test_inverse_exp(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exp1 = Logarithm(Exponent(tree1, tree2), tree1)
        if exp1.validate(variables):
            assert isclose(exp1.evaluate(variables), tree2.evaluate(variables), abs_tol=1e-5)


class TestValidation:
    @given(rpn_expression, var_dict)
    def test_validation_general(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        if tree.validate(variables):
            tree.evaluate(variables)
        else:
            with pytest.raises(Exception):
                assert tree.evaluate(variables)

    @given(rpn_expression, rpn_expression, var_dict)
    def test_validate_product(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        prodtree = Product(tree1, tree2)

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

    @given(rpn_expression, rpn_expression, var_dict)
    def test_validate_division(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        divtree = Division(tree1, tree2)

        tree1_valid = tree1.validate(variables)
        tree2_valid = tree2.validate(variables)
        divtree_valid = divtree.validate(variables)
        if tree1_valid and tree2_valid:
            assert divtree_valid == (tree2.evaluate(variables) != 0)
        else:
            assert not divtree_valid

    @given(rpn_expression, rpn_expression, var_dict)
    def test_validate_exponent(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        exptree = Exponent(tree1, tree2)

        try:
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

    @given(rpn_expression, rpn_expression, var_dict)
    def test_validate_log(self, rpn1, rpn2, variables):
        tree1 = rpn_to_tree(rpn1)
        tree2 = rpn_to_tree(rpn2)
        logtree = Logarithm(tree1, tree2)

        tree1_valid = tree1.validate(variables)
        tree2_valid = tree2.validate(variables)
        logtree_valid = logtree.validate(variables)
        if tree1_valid and tree2_valid:
            tree1_result = tree1.evaluate(variables)
            tree2_result = tree2.evaluate(variables)
            assert logtree_valid == (tree1_result > 0 and tree2_result > 0 and tree2_result != 1)
        else:
            assert not logtree_valid

    @given(rpn_expression, var_dict)
    def test_validate_tan(self, rpn, variables):
        tree1 = rpn_to_tree(rpn)
        tantree = Tangent(tree1)
        if tree1.validate(variables):
            assert tantree.validate(variables) == (tree1.evaluate(variables) % pi != pi / 2)
        else:
            assert not tantree.validate(variables)

    @given(rpn_expression, var_dict)
    def test_validate_asin(self, rpn, variables):
        tree1 = rpn_to_tree(rpn)
        asintree = ArcSine(tree1)
        if tree1.validate(variables):
            assert asintree.validate(variables) == (-1 <= tree1.evaluate(variables) <= 1)
        else:
            assert not asintree.validate(variables)

    @given(rpn_expression, var_dict)
    def test_validate_acos(self, rpn, variables):
        tree1 = rpn_to_tree(rpn)
        acostree = ArcCosine(tree1)
        if tree1.validate(variables):
            assert acostree.validate(variables) == (-1 <= tree1.evaluate(variables) <= 1)
        else:
            assert not acostree.validate(variables)


class TestDerivatives:
    @given(rpn_expression, var_dict)
    def test_derivative_constant_rule(self, rpn, variables):
        f = rpn_to_tree(rpn)
        if f.validate(variables):
            result = f.derivative('').evaluate(variables)
            assert result == 0

    @given(rpn_expression, rpn_expression, var_dict)
    def test_derivative_sum_rule(self, rpn1, rpn2, variables):
        f = rpn_to_tree(rpn1)
        g = rpn_to_tree(rpn2)
        sum_f_g = Addition(f, g)
        if sum_f_g.validate(variables):
            for var in f.dependencies().union(g.dependencies()):
                derivative = sum_f_g.derivative(var)
                if derivative.validate(variables):
                    result = sum_f_g.derivative(var).evaluate(variables)
                    reference = f.derivative(var).evaluate(variables) + g.derivative(var).evaluate(variables)
                    assert result == reference

    @given(rpn_expression, rpn_expression, var_dict)
    def test_derivative_product_rule(self, rpn1, rpn2, variables):
        f = rpn_to_tree(rpn1)
        g = rpn_to_tree(rpn2)
        product_f_g = Product(f, g)
        if product_f_g.validate(variables) and f.validate(variables) and g.validate(variables):
            for var in product_f_g.dependencies():
                derivative = product_f_g.derivative(var)
                if derivative.validate(variables):
                    result = product_f_g.derivative(var).evaluate(variables)
                    reference = f.derivative(var).evaluate(variables) * g.evaluate(variables) \
                                + f.evaluate(variables) * g.derivative(var).evaluate(variables)
                    assert result == reference

    @given(rpn_expression, rpn_expression, var_dict)
    def test_derivative_quotient_rule(self, rpn1, rpn2, variables):
        f = rpn_to_tree(rpn1)
        g = rpn_to_tree(rpn2)
        quotient_f_g = Division(f, g)
        if quotient_f_g.validate(variables):
            for var in f.dependencies().union(g.dependencies()):
                derivative = quotient_f_g.derivative(var)
                if derivative.validate(variables):
                    result = derivative.evaluate(variables)
                    reference = (f.derivative(var).evaluate(variables) * g.evaluate(variables)
                                 - f.evaluate(variables) * g.derivative(var).evaluate(variables)) \
                                / g.evaluate(variables) ** 2
                    assert result == reference

    @given(rpn_expression, st.sampled_from(list(operator_1_in_classes.keys())), var_dict)
    def test_derivative_chain_rule(self, rpn, operator, variables):
        g = rpn_to_tree(rpn)
        h = operator(g)
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

    @given(rpn_expression, var_dict)
    def test_total_derivative(self, rpn, variables):
        tree = rpn_to_tree(rpn)
        total_derivative = tree.total_derivative()
        if tree.validate(variables) and total_derivative.validate(variables):
            result = total_derivative.evaluate(variables)
            reference = sum(tree.derivative(var).evaluate(variables) for var in tree.dependencies())
            assert result == reference

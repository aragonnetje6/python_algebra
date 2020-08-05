"""
Unittests for interpreter using pytest
"""

import pytest
from interpreter import *

rpn_list = ['1 2 +',
            '0 0 +',
            '2 3 * 4 5 * +',
            '5 x 2 ** * 4 x * + 3 +',
            '5 5 x + y 4 / + - 2 * 5 ** x *',
            '5 5 x + y 4 + + + 2 * 5 + x log 5 x 2 ** * 4 x * + 3 + log']

precalc_tuple = [(1, 2, '+'),
                 (0, 0, '+'),
                 ((2, 3, '*'), (4, 5, '*'), '+'),
                 (((5, ('x', 2, '**'), '*'), (4, 'x', '*'), '+'), 3, '+'),
                 ((((5, ((5, 'x', '+'), ('y', 4, '/'), '+'), '-'), 2, '*'), 5, '**'), 'x', '*'),
                 (((((5, ((5, 'x', '+'), ('y', 4, '+'), '+'), '+'), 2, '*'), 5, '+'), 'x', 'log'),
                  (((5, ('x', 2, '**'), '*'), (4, 'x', '*'), '+'), 3, '+'), 'log')]

precalc_var_dict = {'x': 7,
                    'y': 11}


class TestConversions:
    @pytest.mark.parametrize('rpn', rpn_list)
    def test_rpn_to_tuple(self, rpn):
        assert tuple_to_rpn(rpn_to_tuple(rpn)) == rpn

    @pytest.mark.parametrize('entry', precalc_tuple)
    def test_tuple_to_ans(self, entry):
        assert tuple_to_ans(entry, precalc_var_dict) == tuple_to_tree(entry).evaluate(precalc_var_dict)

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_rpn_to_ans(self, rpn):
        assert rpn_to_ans(rpn, precalc_var_dict) == rpn_to_tree(rpn).evaluate(precalc_var_dict)

    @pytest.mark.parametrize('entry', precalc_tuple)
    def test_tuple_to_tree_reversible(self, entry):
        assert tuple_to_tree(entry).tuple() == entry

    @pytest.mark.parametrize('rpn', rpn_list)
    def test_rpn_to_tree_reversible(self, rpn):
        assert rpn_to_tree(rpn).rpn() == rpn

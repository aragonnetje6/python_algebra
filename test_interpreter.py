"""
Unittests for interpreter using pytest
"""

import pytest
from interpreter import *

rpn_list = ['1 2 +',
            '0 0 +',
            '2 3 * 4 5 * +',
            '5 x 2 ^ * 4 x * + 3 +',
            '5 5 x + y 4 / + - 2 * 5 ^ x *',
            '5 5 x + y 4 + + + 2 * 5 + x log 5 x 2 ^ * 4 x * + 3 + log']

precalc_tuple = [(1, 2, '+'),
                 (0, 0, '+'),
                 ((2, 3, '*'), (4, 5, '*'), '+'),
                 (((5, ('x', 2, '^'), '*'), (4, 'x', '*'), '+'), 3, '+'),
                 ((((5, ((5, 'x', '+'), ('y', 4, '/'), '+'), '-'), 2, '*'), 5, '^'), 'x', '*'),
                 (((((5, ((5, 'x', '+'), ('y', 4, '+'), '+'), '+'), 2, '*'), 5, '+'), 'x', 'log'),
                  (((5, ('x', 2, '^'), '*'), (4, 'x', '*'), '+'), 3, '+'), 'log')]

precalc_ans = [3, 0, 26, 276, -19736543.53125, 0.1383251008242367]

precalc_var_dict = {'x': 7,
                    'y': 11}


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

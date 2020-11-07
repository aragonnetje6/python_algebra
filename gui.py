"""
User interface for the tree
"""

from os import system

from IPython import get_ipython
from IPython.display import HTML, display_html
from math_tree import *


def generate_html_doc(expression: 'Node') -> str:
    """generates html code for expression"""
    return '<!DOCTYPE html>' \
           + tag('html',
                 tag('head',
                     tag('title',
                         'python_algebra output'))
                 + tag('body',
                       tag('math',
                           expression.mathml(),
                           'xmlns = "http://www.w3.org/1998/Math/MathML" id = "expr"')))


def display(expression: 'Node') -> None:
    """Generates and opens html representation of expression"""
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
        # noinspection PyTypeChecker
        display_html(
            HTML(
                tag('math',
                    expression.mathml(),
                    'xmlns = "http://www.w3.org/1998/Math/MathML" id = "expr"')))
    else:
        html = generate_html_doc(expression)
        with open('output.html', 'w') as file:
            file.write(html)
        system('output.html')

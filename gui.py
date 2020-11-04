"""
User interface for the tree
"""

from math_tree import Node, tag
from os import system, remove
from time import sleep


def generate_html(expression: Node) -> str:
    """generates html code for expression"""
    return '<!DOCTYPE html>' \
           + tag('html',
                 tag('head',
                     tag('title',
                         'python_algebra output'))
                 + tag('body',
                       tag('math',
                           expression.mathml(),
                           'xmlns = "http://www.w3.org/1998/Math/MathML" id = "expr" title = "Expression"')))


def display(expression: Node):
    """Generates and opens html representation of expression"""
    html = generate_html(expression)
    with open('output.html', 'w') as file:
        file.write(html)
    system('output.html')
    sleep(1)
    remove('output.html')

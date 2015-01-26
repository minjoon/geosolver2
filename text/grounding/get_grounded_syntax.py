from geosolver.text.grounding.get_grounded_syntax_tree import get_grounded_syntax_tree
from geosolver.text.grounding.states import GroundedSyntax

__author__ = 'minjoon'


def get_grounded_syntax(grounded_tokens):
    assert len(grounded_tokens) > 0
    any_grounded_token = grounded_tokens.values()[0]
    syntax = any_grounded_token.syntax
    basic_ontology = any_grounded_token.basic_ontology
    grounded_syntax_trees = {}
    all_tokens = dict(grounded_tokens.items() + syntax.tokens.items())
    for syntax_tree in syntax.syntax_trees.values():
        grounded_syntax_tree = get_grounded_syntax_tree(all_tokens, syntax_tree)
        grounded_syntax_trees[grounded_syntax_tree.rank] = grounded_syntax_tree

    grounded_syntax = GroundedSyntax(syntax, basic_ontology, grounded_tokens, grounded_syntax_trees)
    return grounded_syntax
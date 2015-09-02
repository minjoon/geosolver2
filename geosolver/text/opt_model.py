import itertools
import logging
import numpy as np
from geosolver.grounding.ground_formula import ground_formulas
from geosolver.grounding.states import MatchParse
from geosolver.text.rule import UnaryRule
from geosolver.text.rule_model import CombinedModel
from geosolver.text.semantic_tree import SemanticTreeNode

__author__ = 'minjoon'

class GreedyOptModel(object):

    def objective_function(self, semantic_trees):
        return 0.0

    def optimize(self, semantic_trees, threshold):
        return set()

class TextGreedyOptModel(GreedyOptModel):
    def __init__(self, combined_model):
        assert isinstance(combined_model, CombinedModel)
        self.combined_model = combined_model

    def optimize(self, semantic_trees, threshold):
        selected = set()
        remaining = set(semantic_trees)

        curr_score = self.objective_function(selected)
        next_tree, next_score = self.get_next_tree(selected, remaining)
        if next_tree is None:
            print "No legal next available."
            return set()
        while next_score - curr_score > threshold:
            print "%.2f, %r" % (next_score, next_tree)
            curr_score = next_score
            selected.add(next_tree)
            remaining.discard(next_tree)
            next_tree, next_score = self.get_next_tree(selected, remaining)
            if next_tree is None:
                print "No legal next available."
                break
            next_score = self.objective_function(selected.union([next_tree]))
            if len(selected) > 100:
                raise Exception()
        print ""
        return selected

    def objective_function(self, semantic_trees):
        if len(semantic_trees) == 0:
            return 0.0
        sum_log = sum(np.log(self.combined_model.get_tree_score(tree)) for tree in semantic_trees)
        cov = len(set(tr.span for semantic_tree in semantic_trees for tr in semantic_tree.get_tag_rules()))
        return cov + sum_log

    @staticmethod
    def pairwise_legal(a_tree, b_tree):
        """
        First, return 0 if one of them is self-referencing unary tree
        Do not penalize leaf redundancies
        Penalize everything else

        :param a_tree:
        :param b_tree:
        :return:
        """
        assert isinstance(a_tree, SemanticTreeNode)
        assert isinstance(b_tree, SemanticTreeNode)

        a_spans = set(node.content.span for node in a_tree)
        b_spans = set(node.content.span for node in b_tree)

        for a_span, b_span in itertools.product(a_spans, b_spans):
            a_set = set(range(*a_span))
            b_set = set(range(*b_span))
            if a_set != b_set and (a_set.issubset(b_set) or b_set.issubset(a_set)):
                return False

        common_spans = a_spans.intersection(b_spans)
        for span in common_spans:
            a_tags = a_tree.get_tag_rules_by_span(span)
            b_tags = b_tree.get_tag_rules_by_span(span)
            if a_tags != b_tags:
                return False
        return True


    def get_next_tree(self, selected, remaining):
        d = {each: self.objective_function(selected.union([each])) for each in remaining
             if all(TextGreedyOptModel.pairwise_legal(each, each_selected) for each_selected in selected)}
        if len(d) == 0:
            return None, None
        pair = max(d.iteritems(), key=lambda pair: pair[1])
        return pair


class FullGreedyOptModel(TextGreedyOptModel):
    def __init__(self, combined_model, match_parse):
        super(FullGreedyOptModel, self).__init__(combined_model)
        assert isinstance(match_parse, MatchParse)
        self.match_parse = match_parse
        self.core_parse = match_parse.graph_parse.core_parse
        self.diagram_scores = {}

    def optimize(self, semantic_trees, threshold):
        """
        for t in semantic_trees:
            print "%.3f %r %r" % (self.combined_model.get_tree_score(t), self.get_diagram_score(t), t)
        print ""
        """
        out = super(FullGreedyOptModel, self).optimize(semantic_trees, threshold)
        return out


    def get_diagram_score(self, semantic_tree):
        assert isinstance(semantic_tree, SemanticTreeNode)
        if semantic_tree in self.diagram_scores:
            return self.diagram_scores[semantic_tree]
        formula = semantic_tree.to_formula()
        if formula.has_constant():
            self.diagram_scores[semantic_tree] = None
            return None

        try:
            grounded_formula = ground_formulas(self.match_parse, [formula])[0]
            score = self.match_parse.graph_parse.core_parse.evaluate(grounded_formula).conf
        except Exception as e:
            # logging.exception(e)
            score = None
        self.diagram_scores[semantic_tree] = score
        return score

    def objective_function(self, semantic_trees):
        if len(semantic_trees) == 0:
            return 0.0


        # sum_log = sum(np.log(self.combined_model.get_tree_score(tree)) for tree in semantic_trees)
        cov = len(set(tr.span for semantic_tree in semantic_trees for tr in semantic_tree.get_tag_rules()))
        sum_log = sum(np.log(magic(self.combined_model.get_tree_score(t), self.get_diagram_score(t))) for t in semantic_trees)
        # sum_log = sum(np.log(magic(self.combined_model.get_tree_score(t), diagram_scores[t])) for t in semantic_trees)
        return cov + sum_log

def magic(text_score, diagram_score):
    if diagram_score is None:
        return text_score
    else:
        if diagram_score > text_score:
            return np.mean((text_score, diagram_score))
        else:
            return diagram_score

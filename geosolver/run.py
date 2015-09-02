from cStringIO import StringIO
import logging
import numbers
import sys
import time
from geosolver import geoserver_interface
from geosolver.database.utils import split
from geosolver.diagram.parse_confident_formulas import parse_confident_formulas
from geosolver.diagram.shortcuts import question_to_match_parse
from geosolver.expression.expression_parser import expression_parser
from geosolver.expression.prefix_to_formula import prefix_to_formula
from geosolver.grounding.ground_formula import ground_formulas
from geosolver.grounding.parse_match_formulas import parse_match_formulas
from geosolver.grounding.parse_match_from_known_labels import parse_match_from_known_labels
from geosolver.ontology.ontology_semantics import evaluate, Equals
from geosolver.solver.solve import solve
from geosolver.text.augment_formulas import augment_formulas
from geosolver.text.opt_model import TextGreedyOptModel, GreedyOptModel, FullGreedyOptModel
from geosolver.text.rule_model import CombinedModel
from geosolver.text.run_text import train_semantic_model, questions_to_syntax_parses, train_tag_model
from geosolver.text.semantic_trees_to_text_formula_parse import semantic_trees_to_text_formula_parse
from geosolver.text.annotation_to_semantic_tree import annotation_to_semantic_tree, is_valid_annotation
from geosolver.text.complete_formulas import complete_formulas
from geosolver.text.syntax_parser import stanford_parser
from geosolver.ontology.utils import filter_formulas, reduce_formulas
from geosolver.ontology.utils import flatten_formulas
from geosolver.utils.prep import open_image
import cPickle as pickle

__author__ = 'minjoon'


class SimpleResult(object):
    def __init__(self, id_, error, penalized, correct, duration=-1, message=""):
        assert isinstance(penalized, bool)
        assert isinstance(correct, bool)
        assert isinstance(error, bool)
        assert isinstance(duration, numbers.Real)
        assert isinstance(message, str)
        self.id = id_
        self.penalized = penalized
        self.correct = correct
        self.duration = duration
        self.message = message
        self.error = error

    def __repr__(self):
        return "(e,p,c) = %s, %s, %s" % (self.error, self.penalized, self.correct)

def annotated_unit_test(query):
    """
    Attempts to solve the question with id=id_.
    If the answer is correct, return 'c'
    If the answer is wrong, return 'i'
    If an error occurred, return 'e'

    :param id_:
    :return SimpleResult:
    """
    #myout = StringIO()
    #oldstdout = sys.stdout
    #sys.stdout = myout
    try:
        result = _annotated_unit_test(query)
    except Exception, e:
        logging.error(query)
        logging.exception(e)
        result = SimpleResult(query, True, False, False)
    #sys.stdout = oldstdout
    #message = myout.getvalue()
    #result.message = message
    return result

def _annotated_unit_test(query):
    questions = geoserver_interface.download_questions(query)
    all_annotations = geoserver_interface.download_semantics(query)
    pk, question = questions.items()[0]

    choice_formulas = get_choice_formulas(question)
    label_data = geoserver_interface.download_labels(pk)[pk]
    diagram = open_image(question.diagram_path)
    graph_parse = diagram_to_graph_parse(diagram)
    core_parse = graph_parse.core_parse
    # core_parse.display_points()
    # core_parse.primitive_parse.display_primitives()
    match_parse = parse_match_from_known_labels(graph_parse, label_data)
    match_formulas = parse_match_formulas(match_parse)
    diagram_formulas = parse_confident_formulas(graph_parse)
    all_formulas = match_formulas + diagram_formulas
    for number, sentence_words in question.sentence_words.iteritems():
        syntax_parse = stanford_parser.get_best_syntax_parse(sentence_words)
        annotation_nodes = [annotation_to_semantic_tree(syntax_parse, annotation)
                            for annotation in all_annotations[pk][number].values()]
        expr_formulas = {key: prefix_to_formula(expression_parser.parse_prefix(expression))
                         for key, expression in question.sentence_expressions[number].iteritems()}
        truth_expr_formulas, value_expr_formulas = _separate_expr_formulas(expr_formulas)
        text_formula_parse = semantic_trees_to_text_formula_parse(annotation_nodes)
        completed_formulas = complete_formulas(text_formula_parse)
        grounded_formulas = [ground_formula(match_parse, formula, value_expr_formulas)
                             for formula in completed_formulas+truth_expr_formulas]
        text_formulas = filter_formulas(flatten_formulas(grounded_formulas))
        all_formulas.extend(text_formulas)

    reduced_formulas = reduce_formulas(all_formulas)
    for reduced_formula in reduced_formulas:
        score = evaluate(reduced_formula, core_parse.variable_assignment)
        scores = [evaluate(child, core_parse.variable_assignment) for child in reduced_formula.children]
        print reduced_formula, score, scores
    # core_parse.display_points()

    ans = solve(reduced_formulas, choice_formulas, assignment=core_parse.variable_assignment)
    print "ans:", ans

    if choice_formulas is None:
        attempted = True
        if abs(ans - float(question.answer)) < 0.01:
            correct = True
        else:
            correct = False
    else:
        attempted = True
        c = max(ans.iteritems(), key=lambda pair: pair[1].conf)[0]
        if c == int(question.answer):
            correct = True
        else:
            correct = False

    result = SimpleResult(query, False, attempted, correct)
    return result

def full_unit_test(combined_model, question, label_data):
    """
    Attempts to solve the question with id=id_.
    If the answer is correct, return 'c'
    If the answer is wrong, return 'i'
    If an error occurred, return 'e'

    :param id_:
    :return SimpleResult:
    """
    #myout = StringIO()
    #oldstdout = sys.stdout
    #sys.stdout = myout
    try:
        result = _full_unit_test(combined_model, question, label_data)
    except Exception, e:
        logging.error(question.key)
        logging.exception(e)
        result = SimpleResult(question.key, True, False, False)
    #sys.stdout = oldstdout
    #message = myout.getvalue()
    #result.message = message
    return result

    # graph_parse.core_parse.display_points()

def _full_unit_test(combined_model, question, label_data):
    assert isinstance(combined_model, CombinedModel)

    choice_formulas = get_choice_formulas(question)
    match_parse = question_to_match_parse(question, label_data)
    match_formulas = parse_match_formulas(match_parse)
    graph_parse = match_parse.graph_parse
    core_parse = graph_parse.core_parse
    # core_parse.display_points()
    # core_parse.primitive_parse.display_primitives()

    # opt_model = TextGreedyOptModel(combined_model)

    diagram_formulas = parse_confident_formulas(match_parse.graph_parse)
    all_formulas = match_formulas + diagram_formulas

    opt_model = FullGreedyOptModel(combined_model, match_parse)
    for number, sentence_words in question.sentence_words.iteritems():
        syntax_parse = stanford_parser.get_best_syntax_parse(sentence_words)

        expr_formulas = {key: prefix_to_formula(expression_parser.parse_prefix(expression))
                         for key, expression in question.sentence_expressions[number].iteritems()}
        truth_expr_formulas, value_expr_formulas = _separate_expr_formulas(expr_formulas)

        semantic_forest = opt_model.combined_model.get_semantic_forest(syntax_parse)
        truth_semantic_trees = semantic_forest.get_semantic_trees_by_type("truth")
        is_semantic_trees = semantic_forest.get_semantic_trees_by_type("is")
        cc_trees = set(t for t in semantic_forest.get_semantic_trees_by_type('cc')
                       if opt_model.combined_model.get_tree_score(t) > 0.01)
        for cc_tree in cc_trees:
            print "cc tree:", cc_tree, opt_model.combined_model.get_tree_score(cc_tree)

        bool_semantic_trees = opt_model.optimize(truth_semantic_trees.union(is_semantic_trees), 0)
        # semantic_trees = bool_semantic_trees.union(cc_trees)

        core_formulas = set(t.to_formula() for t in bool_semantic_trees)
        cc_formulas = set(t.to_formula() for t in cc_trees)
        augmented_formulas = augment_formulas(core_formulas)
        completed_formulas = complete_formulas(augmented_formulas, cc_formulas)

        print "completed formulas:"
        for f in completed_formulas: print f
        print ""

        grounded_formulas = ground_formulas(match_parse, completed_formulas+truth_expr_formulas, value_expr_formulas)
        text_formulas = filter_formulas(flatten_formulas(grounded_formulas))
        all_formulas.extend(text_formulas)

    reduced_formulas = all_formulas # reduce_formulas(all_formulas)
    for reduced_formula in reduced_formulas:
        if reduced_formula.is_grounded(core_parse.variable_assignment.keys()):
            score = evaluate(reduced_formula, core_parse.variable_assignment)
            scores = [evaluate(child, core_parse.variable_assignment) for child in reduced_formula.children]
        else:
            score = None
            scores = None
        print reduced_formula, score, scores
    # core_parse.display_points()

    ans = solve(reduced_formulas, choice_formulas, assignment=None)#core_parse.variable_assignment)
    print "ans:", ans

    if choice_formulas is None:
        penalized = False
        if Equals(ans, float(question.answer)).conf > 0.98:
            correct = True
        else:
            correct = False
    else:
        idx, tv = max(ans.iteritems(), key=lambda pair: pair[1].conf)
        if tv.conf > 0.98:
            if idx == int(question.answer):
                correct = True
                penalized = False
            else:
                correct = False
                penalized = True
        else:
            penalized = False
            correct = False

    result = SimpleResult(question.key, False, penalized, correct)
    return result

    # graph_parse.core_parse.display_points()

def _separate_expr_formulas(expr_formulas):
    truth_expr_formulas = []
    value_expr_formulas = {}
    for key, expr_formula in expr_formulas.iteritems():
        if key[1] == 's':
            truth_expr_formulas.append(expr_formula)
        else:
            value_expr_formulas[key] = expr_formula
    return truth_expr_formulas, value_expr_formulas

def get_choice_formulas(question):
    """
    Temporary; will be replaced by text parser
    :param question:
    :return:
    """
    choice_formulas = {}
    for number, choice_expressions in question.choice_expressions.iteritems():
        choice_words = question.choice_words[number]
        if len(choice_expressions) == 1:
            string = choice_expressions.values()[0]
        elif len(choice_expressions) == 0:
            if len(choice_words) == 1:
                string = choice_words.values()[0]
            else:
                continue
                # string = r"\none"
        else:
            return None
        expr_formula = prefix_to_formula(expression_parser.parse_prefix(string))
        choice_formulas[number] = expr_formula
    if len(choice_formulas) == 0:
        return None
    return choice_formulas


def annotated_test():
    ids1 = [963, 968, 969, 971, 973, 974, 977, 985, 990, 993, 995, 1000, 1003, 1004, 1006, 1011, 1014, 1017, 1018, 1020,]
    ids2 = [1025, 1030, 1031, 1032, 1035, 1037, 1038, 1039, 1040, 1042, 1043, 1045, 1047, 1050, 1051, 1052, 1054, 1056, 1058,] # 1027
    ids3 = [1063, 1065, 1067, 1076, 1089, 1095, 1096, 1097, 1099, 1102, 1105, 1106, 1107, 1108, 1110, 1111, 1119, 1120, 1121] # 1103
    ids4 = [1122, 1123, 1124, 1127, 1141, 1142, 1143, 1145, 1146, 1147, 1149, 1150, 1151, 1152, 1070, 1083, 1090, 1092, 1144, 1148]
    ids6 = [997, 1046, 1053]
    ids = ids1 + ids2 + ids3 + ids4 + ids5
    #ids = [973]
    correct = 0
    attempted = 0
    total = len(ids)
    error = 0
    start = time.time()
    for idx, id_ in enumerate(ids):
        print "-"*80
        print "%d/%d complete" % (idx+1, len(ids))
        id_ = str(id_)
        print "-"*80
        print "id: %s" % id_
        result = annotated_unit_test(id_)
        print result.message
        print result
        if result.error:
            error += 1
        if result.attempted:
            attempted += 1
        if result.correct:
            correct += 1
    end = time.time()
    print "-"*80
    print "duration:\t%.1f" % (end - start)

    out = "total:\t\t%d\nattempted:\t%d\ncorrect:\t%d\nerror:\t\t%d" % (total, attempted, correct, error)
    print out

def full_test():
    start = time.time()
    ids1 = [963, 968, 969, 971, 973, 974, 977, 985, 990, 993, 995, 1000, 1003, 1004, 1006, 1014, 1017, 1018, 1020,] #1011
    ids2 = [1025, 1030, 1031, 1032, 1035, 1038, 1039, 1040, 1042, 1043, 1045, 1047, 1050, 1051, 1052, 1054, 1056, 1058,] #1027, 1037
    ids3 = [1063, 1065, 1067, 1076, 1089, 1095, 1096, 1097, 1099, 1102, 1105, 1106, 1107, 1108, 1110, 1111, 1119, 1120, 1121] # 1103
    ids4 = [1122, 1123, 1124, 1127, 1141, 1142, 1143, 1145, 1146, 1147, 1149, 1150, 1151, 1152, 1070, 1083, 1090, 1092, 1144, 1148]
    ids5 = [975, 979, 981, 988, 989, 997, 1005, 1019, 1029, 1044, 1046, 1057, 1059, 1064, 1087, 1104, 1113, 1114, 1129, 1071]
    ids6 = [1100, 1101, 1109, 1140, 1053]
    tr_ids = ids4+ids5+ids6
    te_ids = ids1+ids2+ids3
    # te_ids = [1018]

    load = False

    all_questions = geoserver_interface.download_questions('test')
    if not load:
        all_syntax_parses = questions_to_syntax_parses(all_questions)
        pickle.dump(all_syntax_parses, open('syntax_parses.p', 'wb'))
    else:
        all_syntax_parses = pickle.load(open('syntax_parses.p', 'rb'))
    all_annotations = geoserver_interface.download_semantics()
    all_labels = geoserver_interface.download_labels()

    correct = 0
    penalized = 0
    error = 0
    total = len(te_ids)

    #(te_s, te_a, te_l), (tr_s, tr_a, trl_l) = split([all_syntax_parses, all_annotations, all_labels], 0.7)
    tr_s = {id_: all_syntax_parses[id_] for id_ in tr_ids}
    tr_a = {id_: all_annotations[id_] for id_ in tr_ids}
    te_s = {id_: all_syntax_parses[id_] for id_ in te_ids}

    if not load:
        tm = train_tag_model(all_syntax_parses, all_annotations)
        cm = train_semantic_model(tm, tr_s, tr_a)
        pickle.dump(cm, open('cm.p', 'wb'))
    else:
        cm = pickle.load(open('cm.p', 'rb'))

    print "test ids: %s" % ", ".join(str(k) for k in te_s.keys())
    for idx, (id_, syntax_parse) in enumerate(te_s.iteritems()):
        question = all_questions[id_]
        label = all_labels[id_]
        id_ = str(id_)
        print "-"*80
        print "id: %s" % id_
        result = full_unit_test(cm, question, label)
        print result.message
        print result
        if result.error:
            error += 1
        if result.penalized:
            penalized += 1
        if result.correct:
            correct += 1
        print "-"*80
        print "%d/%d complete, %d correct, %d penalized, %d error" % (idx+1, len(te_s), correct, penalized, error)
    end = time.time()
    print "-"*80
    print "duration:\t%.1f" % (end - start)

    out = "total:\t\t%d\npenalized:\t%d\ncorrect:\t%d\nerror:\t\t%d" % (total, penalized, correct, error)
    print out

if __name__ == "__main__":
    # annotated_test()
    full_test()

from cStringIO import StringIO
import json
import logging
import numbers
from pprint import pprint
import shutil
import sys
import time
import signal
from geosolver import geoserver_interface
from geosolver.database.utils import split
from geosolver.diagram.parse_confident_formulas import parse_confident_formulas
from geosolver.diagram.shortcuts import question_to_match_parse
from geosolver.expression.expression_parser import expression_parser
from geosolver.expression.prefix_to_formula import prefix_to_formula
from geosolver.grounding.ground_formula import ground_formulas
from geosolver.grounding.parse_match_formulas import parse_match_formulas
from geosolver.grounding.parse_match_from_known_labels import parse_match_from_known_labels
from geosolver.ontology.ontology_definitions import FormulaNode, VariableSignature, issubtype
from geosolver.ontology.ontology_semantics import evaluate, Equals
from geosolver.solver.solve import solve
from geosolver.text.augment_formulas import augment_formulas
from geosolver.text.opt_model import TextGreedyOptModel, GreedyOptModel, FullGreedyOptModel
from geosolver.text.rule import TagRule
from geosolver.text.rule_model import CombinedModel
from geosolver.text.run_text import train_semantic_model, questions_to_syntax_parses, train_tag_model
from geosolver.text.semantic_tree import SemanticTreeNode
from geosolver.text.semantic_trees_to_text_formula_parse import semantic_trees_to_text_formula_parse
from geosolver.text.annotation_to_semantic_tree import annotation_to_semantic_tree, is_valid_annotation
from geosolver.text.complete_formulas import complete_formulas
from geosolver.text.syntax_parser import stanford_parser
from geosolver.ontology.utils import filter_formulas, reduce_formulas
from geosolver.ontology.utils import flatten_formulas
from geosolver.utils.prep import open_image
import cPickle as pickle
import os.path
from geosolver.database.states import Question

__author__ = 'minjoon'
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

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
        #print reduced_formula, score, scores
    # core_parse.display_points()

    ans = solve(reduced_formulas, choice_formulas, assignment=core_parse.variable_assignment)
    #print "ans:", ans

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
    maxtime = 2400

    def handler(signum, frame):
        raise Exception("Timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(maxtime)

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

def semantic_tree_to_serialized_entities(match_parse, semantic_tree, sentence_number, value_expr_formulas):
    offset = match_parse.graph_parse.core_parse.image_segment_parse.diagram_image_segment.offset
    formula = semantic_tree.to_formula()
    entities = []
    grounded_formula = ground_formulas(match_parse, [formula], value_expr_formulas)[0]
    zipped_formula = grounded_formula.zip(semantic_tree)
    for zipped_node in zipped_formula:
        formula_node, tree_node = zipped_node.nodes
        if isinstance(formula_node, FormulaNode) and issubtype(formula_node.return_type, 'entity'):
            coords = match_parse.graph_parse.core_parse.evaluate(formula_node)
            if coords is not None:
                coords = offset_coords(coords, tree_node.content.signature.return_type, offset)
                entity = {"content": tree_node.content.serialized(), "coords": serialize_entity(coords),
                          "sentence_number": sentence_number}
                entities.append(entity)
    return entities

def formula_to_serialized_entities(match_parse, formula, tree, sentence_number):
    offset = match_parse.graph_parse.core_parse.image_segment_parse.diagram_image_segment.offset
    grounded_formula = ground_formulas(match_parse, [formula])[0]
    entities = []
    zipped_formula = grounded_formula.zip(tree)
    for zipped_node in zipped_formula:
        formula_node, tree_node = zipped_node.nodes
        if not isinstance(formula_node, FormulaNode):
            continue
        if len(formula_node.children) == 1 and not issubtype(formula_node.return_type, 'entity'):
            formula_node = formula_node.children[0]
        if issubtype(formula_node.return_type, 'entity'):
            coords = match_parse.graph_parse.core_parse.evaluate(formula_node)
            if coords is not None:
                coords = offset_coords(coords, formula_node.return_type, offset)
                content = tree_node.content.serialized()
                content['signature']['return_type'] = formula_node.return_type
                entity = {"content": content, "coords": serialize_entity(coords),
                          "sentence_number": sentence_number}
                entities.append(entity)
    return entities

def offset_point(point, offset):
    return point[0]+offset[0], point[1]+offset[1]

def offset_coords(coords, type_, offset):
    coords = list(coords)
    if issubtype(type_, 'point'):
        coords = offset_point(coords, offset)
    elif issubtype(type_, "line"):
        coords[0] = offset_point(coords[0], offset)
        coords[1] = offset_point(coords[1], offset)
    elif issubtype(type_, 'circle'):
        coords[0] = offset_point(coords[0], offset)
    elif issubtype(type_, 'arc') or issubtype(type_, 'sector'):
        coords[0][0] = offset_point(coords[0][0], offset)
        coords[1] = offset_point(coords[1], offset)
        coords[2] = offset_point(coords[2], offset)
    else:
        coords = [offset_point(point, offset) for point in coords]
    return coords


def serialize_entity(entity):
    try:
        return [serialize_entity(each) for each in entity]
    except:
        return float("%.2f" % entity)

def formula_to_semantic_tree(formula, syntax_parse, span):
    """
    Create dummy semantic tree where each tag's syntax Parse and span is given
    :param formula:
    :param index:
    :return:
    """
    assert isinstance(formula, FormulaNode)
    if issubtype(formula.signature.return_type, 'entity'):
        new_sig = VariableSignature(formula.signature.id, formula.signature.return_type, name='temp')
        tag_rule = TagRule(syntax_parse, span, new_sig)
        return SemanticTreeNode(tag_rule, [])
    tag_rule = TagRule(syntax_parse, span, formula.signature)
    children = [formula_to_semantic_tree(child, syntax_parse, span) for child in formula.children]
    semantic_tree = SemanticTreeNode(tag_rule, children)
    return semantic_tree


demo_path = "../temp/demo"

def _full_unit_test(combined_model, question, label_data):
    assert isinstance(combined_model, CombinedModel)

    base_path = os.path.join(demo_path, str(question.key))
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    question_path = os.path.join(base_path, 'question.json')
    text_parse_path = os.path.join(base_path, 'text_parse.json')
    diagram_parse_path = os.path.join(base_path, 'diagram_parse.json')
    optimized_path = os.path.join(base_path, 'optimized.json')
    entity_list_path = os.path.join(base_path, 'entity_map.json')
    diagram_path = os.path.join(base_path, 'diagram.png')
    solution_path = os.path.join(base_path, 'solution.json')
    #shutil.copy(question.diagram_path, diagram_path)
    text_parse_list = []
    diagram_parse_list = []
    optimized_list = []
    entity_list = []
    solution = ""
    #json.dump(question._asdict(), open(question_path, 'wb'))
 
    choice_formulas = get_choice_formulas(question)
    match_parse = question_to_match_parse(question, label_data)
    print "***"
    print question
    print label_data
    print match_parse
    match_formulas = parse_match_formulas(match_parse)
    graph_parse = match_parse.graph_parse
    core_parse = graph_parse.core_parse
    # core_parse.display_points()
    # core_parse.primitive_parse.display_primitives()

    # opt_model = TextGreedyOptModel(combined_model)
 
    diagram_formulas = parse_confident_formulas(match_parse.graph_parse)
    all_formulas = set(match_formulas + diagram_formulas)
 
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
        #for cc_tree in cc_trees:
        #    print "cc tree:", cc_tree, opt_model.combined_model.get_tree_score(cc_tree)

        bool_semantic_trees = opt_model.optimize(truth_semantic_trees.union(is_semantic_trees), 0, cc_trees)
        # semantic_trees = bool_semantic_trees.union(cc_trees)
 
        for t in truth_semantic_trees.union(is_semantic_trees).union(cc_trees):
            text_parse_list.append({'simple': t.simple_repr(), 'tree': t.serialized(), 'sentence_number': number,
                                    'score': opt_model.combined_model.get_tree_score(t)})
            diagram_score = opt_model.get_diagram_score(t.to_formula(), cc_trees)
            if diagram_score is not None:
                diagram_parse_list.append({'simple': t.simple_repr(), 'tree': t.serialized(), 'sentence_number': number,
                                           'score': diagram_score})

            local_entities = semantic_tree_to_serialized_entities(match_parse, t, number, value_expr_formulas)
            entity_list.extend(local_entities)

        for t in bool_semantic_trees:
            optimized_list.append({'simple': t.simple_repr(), 'tree': t.serialized(), 'sentence_number': number,
                                    'score': opt_model.get_magic_score(t, cc_trees)})

        for key, f in expr_formulas.iteritems():
            if key.startswith("v"):
                pass
            index = (i for i, word in sentence_words.iteritems() if word == key).next()
            tree = formula_to_semantic_tree(f, syntax_parse, (index, index+1))
            #print "f and t:", f, tree
            text_parse_list.append({'simple': f.simple_repr(), 'tree': tree.serialized(), 'sentence_number': number, 'score': 1.0})
            optimized_list.append({'simple': f.simple_repr(), 'tree': tree.serialized(), 'sentence_number': number, 'score': 1.0})

            local_entities = formula_to_serialized_entities(match_parse, f, tree, number)
            #print "local entities:", local_entities
            entity_list.extend(local_entities)



        core_formulas = set(t.to_formula() for t in bool_semantic_trees)
        cc_formulas = set(t.to_formula() for t in cc_trees)
        augmented_formulas = augment_formulas(core_formulas)
        completed_formulas = complete_formulas(augmented_formulas, cc_formulas)

        #print "completed formulas:"
        #for f in completed_formulas: print f
        #print ""

        grounded_formulas = ground_formulas(match_parse, completed_formulas+truth_expr_formulas, value_expr_formulas)
        text_formulas = filter_formulas(flatten_formulas(grounded_formulas))
        all_formulas = all_formulas.union(text_formulas)

    reduced_formulas = all_formulas # reduce_formulas(all_formulas)
    for reduced_formula in reduced_formulas:
        if reduced_formula.is_grounded(core_parse.variable_assignment.keys()):
            score = evaluate(reduced_formula, core_parse.variable_assignment)
            scores = [evaluate(child, core_parse.variable_assignment) for child in reduced_formula.children]
        else:
            score = None
            scores = None
        solution += repr(reduced_formula) + '\n'
        #print reduced_formula, score, scores
    solution = solution.rstrip()
    # core_parse.display_points()

    #json.dump(diagram_parse_list, open(diagram_parse_path, 'wb'))
    #json.dump(optimized_list, open(optimized_path, 'wb'))
    #json.dump(text_parse_list, open(text_parse_path, 'wb'))
    #json.dump(entity_list, open(entity_list_path, 'wb'))
    #json.dump(solution, open(solution_path, 'wb'))


    #print "Solving..."
    ans = solve(reduced_formulas, choice_formulas, assignment=None)#core_parse.variable_assignment)
    #print "ans:", ans


    if choice_formulas is None:
        penalized = False
        if Equals(ans, float(question.answer)).conf > 0.98:
            correct = True
        else:
            correct = False
    else:
        idx, tv = max(ans.iteritems(), key=lambda pair: pair[1].conf)
        if tv.conf > 0.98:
            if idx == int(float(question.answer)):
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


def pretrain():
    all_syntax_parses = pickle.load(open('syntax_parses.p', 'rb'))
    all_annotations = geoserver_interface.download_semantics()
    all_labels = geoserver_interface.download_labels()
    ids4 = [1122, 1123, 1124, 1127, 1141, 1142, 1143, 1145, 1146, 1147, 1149, 1150, 1151, 1152, 1070, 1083, 1090, 1092, 1144, 1148]
    ids5 = [975, 979, 981, 988, 989, 997, 1005, 1019, 1029, 1044, 1046, 1057, 1059, 1064, 1087, 1104, 1114, 1071] #1113,1129
    ids6 = [1109, 1140, 1053] #1100, 1101
    tr_ids = ids4 + ids5 + ids6
    tr_s = {id_: all_syntax_parses[id_] for id_ in tr_ids}
    tr_a = {id_: all_annotations[id_] for id_ in tr_ids}
    tm = train_tag_model(tr_s, tr_a)
    cm = train_semantic_model(tm, tr_s, tr_a)
    pickle.dump(cm, open('cm.p', 'wb'))

def normalize_label(label):
    import re
    label = re.sub(ur'\u221a([0-9A-Za-z]+)', ur'\\sqrt{\1}', label)  
    label = re.sub(r'(.*)([0-9])([A-Za-z])(.*)', r'\1\2*\3\4', label) # replace implicit mult with explicit mult
    label = re.sub(r'([0-9A-Za-z])\\sqrt', r'\1*\\sqrt', label) 
    #label = re.sub(r'\\sqrt\{(.*)\}', r'\\sqrt \1', label) 
     
    #labelgroups = re.search(r'(.*)\\sqrt\{(.*)\}(.*)', label)
    #if labelgroups != None:
    #    import math
    #    sqrt = '%.14f' % math.sqrt(float(labelgroups.groups(0)[1]))
    #    label = re.sub(r'\\sqrt\{(.*)\}', sqrt, label)
    #    labelgroups = re.match(r'(.+)\*(.+)', label)
    #    product = '%.14f' % (float(labelgroups.groups(0)[0]) * float(labelgroups.groups(0)[1]))
    #    label = product
 
    label = label.replace(u"\xb0", "")    
    try:
        udata=label.decode("utf-8")
        return udata.encode("ascii","ignore")
    except UnicodeEncodeError:
        return label

def normalize_dictionary(d):
    result = dict()
    for key in d:
        result[normalize_label(key)] = normalize_label(d[key])
    return result

def normalize_dictionary_values(d):
    result = dict()
    for key in d:
        result[key] = normalize_label(d[key])
    return result

def convert_choice_key(choice_key):
    result = choice_key
    if choice_key == 'A':
        result = 1
    elif choice_key == 'B':
        result = 2
    elif choice_key == 'C':
        result = 3
    elif choice_key == 'D':
        result = 4
    elif choice_key == 'E':
        result = 5
    return result

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def preprocess_question(ques):
    import re    
    ques = re.sub(r'\\\((.+?)\\\)', r'\1', ques)  
    ques = re.sub(r'\\triangle', r'triangle', ques)  
    ques = re.sub(r'\\overline\{(.+?)\}', r'\1', ques)  
    ques = re.sub(r'\\frac\{(.+?[\+\-\*\^/].+?)\}\{(.+?)\}', r'(\1) / \2', ques)  
    ques = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1 / \2', ques)  
    ques = re.sub(r'([A-Za-z0-9])\\pi', r'\1*\\pi', ques)  
    ques = re.sub(r'_\{([0-9])\}', r'\1', ques)  
    return ques

def construct_question(key, text, diagram_path, answer, choices):
    from geosolver.utils.prep import paragraph_to_sentences
    from geosolver.utils.prep import sentence_to_words_statements_values
    text = preprocess_question(text)
    sents = paragraph_to_sentences(text)
    sentence_words = dict()
    sentence_expressions = dict()
    for sent in sents:
        (sent_words, sent_exprs, x) = sentence_to_words_statements_values(sents[sent])
        sent_exprs.update(x)
        sentence_words[sent] = normalize_dictionary_values(sent_words)
        sentence_expressions[sent] = normalize_dictionary_values(sent_exprs)
    choice_words = dict()
    choice_expressions = dict()
    #choices = normalize_dictionary_values(choices)
    newchoices = dict()
    for choice in choices:
        choice_text = normalize_label(preprocess_question(choices[choice]))
        (choice_wds, choice_exprs, x) = sentence_to_words_statements_values(choice_text)
        choice_exprs.update(x)
        #if not is_number(choices[choice]):
        #    choice_words[convert_choice_key(choice)] = {0: '@v_0'}
        #    choice_expressions[convert_choice_key(choice)] = {'@v_0': choices[choice]}        
        #else:
        choice_words[convert_choice_key(choice)] = normalize_dictionary_values(choice_wds)
        choice_expressions[convert_choice_key(choice)] = normalize_dictionary_values(choice_exprs)
        newchoices[convert_choice_key(choice)] = choice_text
    choices = newchoices
    answer = convert_choice_key(answer)
    question = Question(key=key, text=text, sentence_words=sentence_words, sentence_expressions=sentence_expressions,
                        diagram_path=diagram_path, choice_words=choice_words, choice_expressions=choice_expressions,
                        answer=answer, choices=choices)
    return question


def display_questions(question_ids):
    for question_id in question_ids:
        all_questions = geoserver_interface.download_questions(question_id)
        question = all_questions[question_id]
        print question.text
        print question.sentence_words
        print question.sentence_expressions
    #all_labels = geoserver_interface.download_labels()
    #label = all_labels[question_id]
        
def get_seo_practice_questions():
    euclid_path = '/Users/markhopkins/Projects/ai2/euclid2/'
    geometry_path = euclid_path + 'data/private/GeometryQuestions.json'
    diagrams_path = euclid_path + 'data/private/diagrams/MathSATImages/'
    import json
    with open(geometry_path) as data_file:    
        geometry_data = json.load(data_file)
    seo_practice = filter(lambda x: 'seo-practice' in x['tags'], geometry_data)
    converted = []
    for datum in seo_practice:
        if 'choices' in datum:
            choices = datum['choices']
        else:
            choices = {}
        question = construct_question(
            key=datum['originalQuestionNumber'],
            text=datum['question'],
            diagram_path=diagrams_path + datum['diagramRef'],
            answer=datum['answer'],
            choices=choices
        )
        converted.append(question)
    return converted

def get_seo_official_questions():
    euclid_path = '/Users/markhopkins/Projects/ai2/euclid2/'
    geometry_path = euclid_path + 'data/private/GeometryQuestions.json'
    diagrams_path = euclid_path + 'data/private/diagrams/MathSATImages/'
    import json
    with open(geometry_path) as data_file:    
        geometry_data = json.load(data_file)
    seo_practice = filter(lambda x: 'seo-official' in x['tags'], geometry_data)
    converted = []
    for datum in seo_practice:
        if 'choices' in datum:
            choices = datum['choices']
        else:
            choices = {}
        question = construct_question(
            key=datum['originalQuestionNumber'],
            text=datum['question'],
            diagram_path=diagrams_path + datum['diagramRef'],
            answer=datum['answer'],
            choices=choices
        )
        converted.append(question)
    return converted
        

def get_kaplan_geometry_questions():
    euclid_path = '/Users/markhopkins/Projects/ai2/euclid2/'
    geometry_path = euclid_path + 'data/private/SATQuestionsBatch1.json'
    diagrams_path = euclid_path + 'data/private/diagrams/MathSATImages/'
    import json
    with open(geometry_path) as data_file:    
        geometry_data = json.load(data_file)
    seo_practice = filter(lambda x: 'geometry' in x['tags'] and x['exam'].startswith('Kaplan'), geometry_data)
    converted = []
    for datum in seo_practice:
        if 'choices' in datum:
            choices = datum['choices']
        else:
            choices = {}
        if 'diagramRef' in datum:
            question = construct_question(
                key=datum['id'],
                text=datum['question'],
                diagram_path=diagrams_path + datum['diagramRef'],
                answer=datum['answer'],
                choices=choices
            )
            converted.append(question)        
    return converted
        
        
def get_mcgrawhill_geometry_questions():
    euclid_path = '/Users/markhopkins/Projects/ai2/euclid2/'
    geometry_path = euclid_path + 'data/private/SATQuestionsBatch1.json'
    diagrams_path = euclid_path + 'data/private/diagrams/MathSATImages/'
    import json
    with open(geometry_path) as data_file:    
        geometry_data = json.load(data_file)
    seo_practice = filter(lambda x: 'geometry' in x['tags'] and x['exam'].startswith('McGraw'), geometry_data)
    converted = []
    for datum in seo_practice:
        if 'choices' in datum:
            choices = datum['choices']
        else:
            choices = {}
        if 'diagramRef' in datum:
            question = construct_question(
                key=datum['id'],
                text=datum['question'],
                diagram_path=diagrams_path + datum['diagramRef'],
                answer=datum['answer'],
                choices=choices
            )
            converted.append(question)        
    return converted
        
def get_official_geometry_questions():
    euclid_path = '/Users/markhopkins/Projects/ai2/euclid2/'
    geometry_path = euclid_path + 'data/private/SATQuestionsBatch1.json'
    diagrams_path = euclid_path + 'data/private/diagrams/MathSATImages/'
    import json
    with open(geometry_path) as data_file:    
        geometry_data = json.load(data_file)
    seo_practice = filter(lambda x: 'geometry' in x['tags'] and x['exam'].startswith('Official'), geometry_data)
    converted = []
    for datum in seo_practice:
        if 'choices' in datum:
            choices = datum['choices']
        else:
            choices = {}
        if 'diagramRef' in datum:
            question = construct_question(
                key=datum['id'],
                text=datum['question'],
                diagram_path=diagrams_path + datum['diagramRef'],
                answer=datum['answer'],
                choices=choices
            )
            converted.append(question)        
    return converted
        

def get_annotated_kaplan_questions():
    import ntpath
    questions = get_kaplan_geometry_questions()
    all_labels = get_annotations()
    annotated = [ques for ques in questions if ntpath.basename(ques.diagram_path) in all_labels]
    return annotated
 
def get_annotated_mcgrawhill_questions():
    import ntpath
    questions = get_mcgrawhill_geometry_questions()
    print(len(questions))
    all_labels = get_annotations()
    annotated = [ques for ques in questions if ntpath.basename(ques.diagram_path) in all_labels]
    return annotated
 
def get_annotated_official_questions():
    import ntpath
    questions = get_official_geometry_questions()
    print(len(questions))
    all_labels = get_annotations()
    annotated = [ques for ques in questions if ntpath.basename(ques.diagram_path) in all_labels]
    return annotated
           

        
def solve_single_question(question_id):
    start = time.time()
    print "-"*80
    print "id: %s" % str(question_id)
    all_questions = geoserver_interface.download_questions(question_id)
    all_labels = geoserver_interface.download_labels()
    cm = pickle.load(open('cm.p', 'rb')) 
    orig_question = all_questions[question_id]
    label = all_labels[question_id]
    print "text: %s" % str(orig_question.text)
    print "label: %s" % str(label)
    print "orig question: %s" % str(orig_question)
    question = construct_question(
        orig_question.key,
        orig_question.text,
        orig_question.diagram_path,
        orig_question.answer,
        orig_question.choices)
    print "modd question: %s" % str(question)
    result = full_unit_test(cm, question, label)
    end = time.time()
    print "duration:\t%.1f" % (end - start)
    print result
    retval = 0
    if result.error:
        retval = 0
        print ("ERROR!!!")
    if result.penalized:
        retval = -1
        print ("PENALIZED!!!")
    if result.correct:
        retval = 1
        print ("CORRECT!!!")
    print "-"*80
    return retval
        
 


def convert_textlabel(label):
    p1x = label['boundingBox']['p1']['x']
    p1y = label['boundingBox']['p1']['y']
    p2x = label['boundingBox']['p2']['x']
    p2y = label['boundingBox']['p2']['y']
    labelx = (p1x + p2x) / 2
    labely = (p1y + p2y) / 2
    labeltext=normalize_label(label['text'])
    labeltype = 'ignore'
    if label['labelType'] == 'rightangle':
        labeltype = 'angle angle'
        labeltext = '90'
    elif label['labelType'] == 'angle':
        labeltype = 'angle angle'
    elif label['labelType'] == 'point':
        labeltype = 'point'
    elif label['labelType'] == 'linelength':
        labeltype = 'length line'
    elif label['labelType'] == 'line-length':
        labeltype = 'length line'
    elif label['labelType'] == 'line':
        labeltype = 'line'
    elif label['labelType'] == 'circle':
        labeltype = 'point'
    return {'x': labelx, 'y': labely, 'label': labeltext, 'type': labeltype}
    
def convert_textlabels(labels):
    return filter(lambda x: x['type'] != 'ignore', [convert_textlabel(label) for label in labels])

def get_annotations():
    euclid_path = '/Users/markhopkins/Projects/ai2/euclid/'
    annotations_path = euclid_path + 'data/private/diagrams/golden/annotations.json'
    #annotations_path = '/Users/markhopkins/Projects/ai2/geosolver/annotations.json'

    import json
    with open(annotations_path) as data_file:    
        annotations_data = json.load(data_file)
    return {datum['diagramRef'] + '.png': convert_textlabels(datum['textLabels']) for datum in annotations_data}

def solve_individual_question(question):
    start = time.time()
    print "-"*80
    print "id: %s" % str(question.key)
    all_labels = get_annotations()
    cm = pickle.load(open('cm.p', 'rb')) 
    import ntpath
    label = all_labels[ntpath.basename(question.diagram_path)]
    print "text: %s" % str(question.text)
    print "question: %s" % str(question)
    print "label: %s" % str(label)
    result = full_unit_test(cm, question, label)
    end = time.time()
    print "duration:\t%.1f" % (end - start)
    print result
    retval = 0
    if result.error:
        retval = 0
        print ("ERROR!!!")
    if result.penalized:
        retval = -1
        print ("PENALIZED!!!")
    if result.correct:
        retval = 1
        print ("CORRECT!!!")
    print "-"*80
    return retval

def solve_questions(question_ids):
    results = [solve_single_question(question_id) for question_id in question_ids]
    print "-"*80
    print results
    correct = sum([result for result in results if result == 1])
    print "Num correct: %s" % str(correct)
    print "-"*80


def solve_seo_practice_questions(i, j):
    questions = get_seo_practice_questions()
    results = [solve_individual_question(question) for question in questions[i:j]]
    correct = sum([result for result in results if result == 1])
    print "batch ending with %s" % str(j)
    print results
    print "Num correct: %s" % str(correct)
    print "-"*80


def solve_seo_official_questions(i, j):
    questions = get_seo_official_questions()
    results = [solve_individual_question(question) for question in questions[i:j]]
    correct = sum([result for result in results if result == 1])
    print "batch ending with %s" % str(j)
    print results
    print "Num correct: %s" % str(correct)
    print "-"*80
    
def solve_kaplan_questions(i, j):
    questions = get_annotated_kaplan_questions()
    print "num annotated kaplan questions: %s" % str(len(questions))
    results = [solve_individual_question(question) for question in questions[i:j]]
    correct = sum([result for result in results if result == 1])
    question_ids = [question.key for question in questions[i:j]]
        
    print "batch ending with %s" % str(j)
    print results
    print question_ids
    print "Num correct: %s" % str(correct)
    print "-"*80
    
def solve_mcgrawhill_questions(i, j):
    questions = get_annotated_mcgrawhill_questions()
    print "num annotated mcgrawhill questions: %s" % str(len(questions))
    results = [solve_individual_question(question) for question in questions[i:j]]
    correct = sum([result for result in results if result == 1])
    print "batch ending with %s" % str(j)
    print results
    print "Num correct: %s" % str(correct)
    print "-"*80
    
def solve_official_questions(i, j):
    questions = get_annotated_official_questions()
    print "num annotated official questions: %s" % str(len(questions))
    results = [solve_individual_question(question) for question in questions[i:j]]
    question_ids = [question.key for question in questions[i:j]]
    correct = sum([result for result in results if result == 1])
    print "batch ending with %s" % str(j)
    print question_ids
    print results
    print "Num correct: %s" % str(correct)
    print "-"*80
    
def solve_correct_official_questions():
    #correct_ids = [0, 2, 6, 18, 23, 32, 38, 42, 43, 45, 46, 53, 58]
    correct_ids = [2]
    questions = get_annotated_official_questions()
    print "num annotated official questions: %s" % str(len(questions))
    results = [solve_individual_question(questions[qid]) for qid in correct_ids]
    question_ids = [questions[qid].key for qid in correct_ids]
    correct = sum([result for result in results if result == 1])
    print question_ids
    print results
    print "Num correct: %s" % str(correct)
    print "-"*80
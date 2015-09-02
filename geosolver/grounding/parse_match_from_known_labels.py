import logging

import numpy as np

from geosolver.diagram.get_instances import get_all_instances
from geosolver.diagram.states import GraphParse
from geosolver.grounding.label_distances import label_distance_to_line, label_distance_to_point, label_distance_to_arc, \
    label_distance_to_angle
from geosolver.grounding.states import MatchParse
from geosolver.ontology.instantiator_definitions import instantiators
from geosolver.ontology.ontology_definitions import FormulaNode, signatures, issubtype
from geosolver.ontology.ontology_semantics import evaluate

__author__ = 'minjoon'


def parse_match_from_known_labels(graph_parse, known_labels):
    assert isinstance(graph_parse, GraphParse)
    match_dict = {}
    point_key_dict = {}
    offset = graph_parse.image_segment_parse.diagram_image_segment.offset
    for idx, d in enumerate(known_labels):
        label = d['label']
        x = d['x'] - offset[0]
        y = d['y'] - offset[1]
        label_point = instantiators['point'](x, y)
        type_ = d['type']
        arr = type_.split(' ')
        if len(arr) > 1:
            type_ = arr[-1]

        # Find closest type_ instance's key in graph_parse
        instances = get_all_instances(graph_parse, type_)
        if len(instances) == 0:
            logging.error("no instance found of type %s" % type_)
            continue

        if len(arr) > 1 and type_ == 'line' and arr[0] == 'length':
            distances = [(key, label_distance_to_line(label_point, instance, True)) for key, instance in instances.iteritems()]
        elif type_ == 'line':
            distances = [(key, label_distance_to_line(label_point, instance, False)) for key, instance in instances.iteritems()]
        elif type_ == 'point':
            distances = [(key, label_distance_to_point(label_point, instance)) for key, instance in instances.iteritems()]
        elif type_ == 'arc':
            distances = [(key, label_distance_to_arc(label_point, instance)) for key, instance in instances.iteritems()]
        elif type_ == 'angle':
            # filter subangles
            # instances = {key: value for key, value in instances.iteritems() if all(x == value or not is_subangle(x, value) for x in instances.values())}
            distances = [(key, label_distance_to_angle(label_point, instance)) for key, instance in instances.iteritems()]

        # Then use the key to get corresponding variable in general graph
        # Wrap the general instance in function nod3. If there are extra prefixes, add these as well the formula
        argmin_key = min(distances, key=lambda pair: pair[1])[0]
        if type_ == 'line':
            a_key, b_key = argmin_key
            a_point = graph_parse.point_variables[a_key]
            b_point = graph_parse.point_variables[b_key]
            formula = FormulaNode(signatures['Line'], [a_point, b_point])
            if len(arr) > 1 and arr[0] == 'length':
                formula = FormulaNode(signatures['LengthOf'], [formula])
        elif type_ == 'point':
            formula = graph_parse.point_variables[argmin_key]
            point_key_dict[label] = argmin_key
        elif type_ == 'angle':
            a_key, b_key, c_key = argmin_key
            a_point = graph_parse.point_variables[a_key]
            b_point = graph_parse.point_variables[b_key]
            c_point = graph_parse.point_variables[c_key]
            formula = FormulaNode(signatures['Angle'], [a_point, b_point, c_point])
            if len(arr) > 1 and arr[0] == 'angle':
                formula = FormulaNode(signatures['MeasureOf'], [formula])
                formula = FormulaNode(signatures['Div'], [formula, FormulaNode(signatures['Degree'], [])])
        elif type_ == 'arc':
            (center_key, radius_key), a_key, b_key = argmin_key
            center_point = graph_parse.point_variables[center_key]
            radius = graph_parse.radius_variables[center_key][radius_key]
            circle = FormulaNode(signatures['Circle'], [center_point, radius])
            a_point = graph_parse.point_variables[a_key]
            b_point = graph_parse.point_variables[b_key]
            formula = FormulaNode(signatures['Arc'], [circle, a_point, b_point])
        if label not in match_dict:
            match_dict[label] = []
        elif issubtype(formula.return_type, 'entity'):
            raise Exception()
        match_dict[label].append(formula)

    match_parse = MatchParse(graph_parse, match_dict, point_key_dict)
    return match_parse

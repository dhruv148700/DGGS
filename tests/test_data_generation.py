from dependency_graph import DependencyGraph
from data_utils import create_label_vector, reindex_nodes, create_hetero_graph
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent


def test_reindex_nodes():
    file = f"{TEST_DIR}/framework1.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(file)
    dep_graph.create_dependency_graph()
    (rules_mapping, assmpt_mapping, non_assmpt_mapping) = reindex_nodes(dep_graph)
    
    assert len(rules_mapping) == len(dep_graph.rules)
    assert len(assmpt_mapping) == len(dep_graph.assumptions)
    assert len(non_assmpt_mapping) == len(dep_graph.non_assumptions)

def test_create_label_vector():
    file = f"{TEST_DIR}/framework1.txt"
    output_file = f"{TEST_DIR}/output_framework1.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(file)
    dep_graph.create_dependency_graph()

    (rules_mapping, assmpt_mapping, non_assmpt_mapping) = reindex_nodes(dep_graph)
    label_vector = create_label_vector(output_file, assmpt_mapping)

    assert len(label_vector) == 5
    assert label_vector[assmpt_mapping['1']] == 1
    assert label_vector[assmpt_mapping['5']] == 1
    assert label_vector[assmpt_mapping['3']] == 1

def test_create_hetero_graph1():
    file = f"{TEST_DIR}/framework1.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(file)
    dep_graph.create_dependency_graph()

    (rules_mapping, assmpt_mapping, non_assmpt_mapping) = reindex_nodes(dep_graph)
    hetero_graph = create_hetero_graph(dep_graph.graph, rules_mapping, assmpt_mapping, non_assmpt_mapping, True)
    assert set(hetero_graph.canonical_etypes) == {
        ('assmpt', 'attacks', 'assmpt'), ('assmpt', 'supports', 'assmpt'), 
        ('assmpt', 'supports', 'rule'), ('non_assmpt', 'attacks', 'assmpt'), 
        ('non_assmpt', 'supports', 'non_assmpt'), ('non_assmpt', 'supports', 'rule'), 
        ('rule', 'derives', 'assmpt'), ('rule', 'derives', 'non_assmpt'), 
        ('rule', 'supports', 'rule')
    }
    assert len(hetero_graph.edges(etype=('assmpt', 'attacks', 'assmpt'))[0]) == 0 
    assert len(hetero_graph.edges(etype=('non_assmpt', 'attacks', 'assmpt'))[0]) == len(dep_graph.contrary)
    assert len(hetero_graph.edges(etype=('assmpt', 'supports', 'assmpt'))[0]) == len(dep_graph.assumptions)
    assert len(hetero_graph.edges(etype=('assmpt', 'supports', 'rule'))[0]) == len(dep_graph.rules)
    assert len(hetero_graph.edges(etype=('non_assmpt', 'supports', 'rule'))[0]) == 0
    assert len(hetero_graph.edges(etype=('non_assmpt', 'supports', 'non_assmpt'))[0]) == len(dep_graph.non_assumptions)
    assert len(hetero_graph.edges(etype=('rule', 'supports', 'rule'))[0]) == len(dep_graph.rules)
    assert len(hetero_graph.edges(etype=('rule', 'derives', 'assmpt'))[0]) == 0
    assert len(hetero_graph.edges(etype=('rule', 'derives', 'non_assmpt'))[0]) == len(dep_graph.rules)

# def test_create_hetero_graph4():
#     file = f"{TEST_DIR}/framework4.txt"
#     dep_graph = DependencyGraph()
#     dep_graph.create_from_file(file)
#     dep_graph.create_dependency_graph()

#     (graph, rules_mapping, atoms_mapping) = reindex_nodes(dep_graph.graph)
#     hetero_graph = create_hetero_graph(graph)
#     assert hetero_graph.ntypes == ['a', 'r']
#     assert hetero_graph.canonical_etypes == [('a', 'attacks', 'a'), ('a', 'supports', 'a'), ('a', 'supports', 'r'), ('r', 'derives', 'a'), ('r', 'supports', 'r')]
#     assert len(hetero_graph.edges(etype='derives')[0]) == len(dep_graph.rules) 
#     assert len(hetero_graph.edges(etype='attacks')[0]) == len(dep_graph.contrary)
#     assert len(hetero_graph.edges(etype=('r', 'supports', 'r'))[0]) == len(dep_graph.rules)
#     assert len(hetero_graph.edges(etype=('a', 'supports', 'a'))[0]) == len(atoms_mapping)
#     body_elems=[elem for value in dep_graph.rules.values() for elem in value[1]]
#     assert len(hetero_graph.edges(etype=('a', 'supports', 'r'))[0]) == len(body_elems)


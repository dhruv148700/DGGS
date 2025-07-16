from dependency_graph.dependency_graph import DependencyGraph
from pathlib import Path
from data_generation import reindex_nodes

TEST_DIR = Path(__file__).resolve().parent

def test_create_from_file():
    file = f"{TEST_DIR}/framework4.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(file)
    assert (dep_graph.assumptions == {'b'})
    contrary = {'b': ["b'"]}
    assert (dep_graph.contrary == contrary)
    rules = [('p',['q','a']), ('q',['b','c']), ('q',['d']), ('p',['q','d'])]
    for key, value in dep_graph.rules.items():
        assert value in rules 

def test_dependency_graph():
    file = f"{TEST_DIR}/framework4.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(file)
    dep_graph.create_dependency_graph()
    assert (set(dep_graph.graph.nodes()) == {'b', "b'", 'a', 'p', 'q', 'c', 'd', 'r1', 'r2', 'r3', 'r4'})
    graph_edges = list(dep_graph.graph.edges(data=True))
    expected_edges = [
        ('a', 'r1', {'label': '+'}), 
        ('q', 'r1', {'label': '+'}),
        ('r1', 'p', {'label': 'd'}),
        ('b', 'r2', {'label': '+'}),
        ('c', 'r2', {'label': '+'}),
        ('r2', 'q', {'label': 'd'}),
        ('q', 'r4', {'label': '+'}),
        ('d', 'r4', {'label': '+'}),
        ('r4', 'p', {'label': 'd'}),
        ('d', 'r3', {'label': '+'}),
        ('r3', 'q', {'label': 'd'}),
        ("b'", 'b', {'label': '-'}),
    ]
    assert len(graph_edges) == len(expected_edges)
    for edge in expected_edges:
        assert edge in graph_edges

def test_larger_dependency_graph():
    file = f"{TEST_DIR}/framework1.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(file)
    dep_graph.create_dependency_graph()
    assert (set(dep_graph.graph.nodes()) == {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'r6', 'r5', 'r4', 'r3', 'r1', 'r2'})
    graph_edges = list(dep_graph.graph.edges(data=True))
    expected_edges = [
        ('1', 'r1', {'label': '+'}),
        ('r1', '9', {'label': 'd'}), 
        ('1', 'r2', {'label': '+'}),
        ('r2', '7', {'label': 'd'}),
        ('2', 'r3', {'label': '+'}),
        ('r3', '8', {'label': 'd'}),
        ('3', 'r4', {'label': '+'}),
        ('r4', '6', {'label': 'd'}),
        ('4', 'r5', {'label': '+'}),
        ('r5', '10', {'label': 'd'}),
        ('5', 'r6', {'label': '+'}),
        ('r6', '6', {'label': 'd'}),
        ('6', '1', {'label': '-'}),
        ('7', '2', {'label': '-'}),
        ('8', '3', {'label': '-'}),
        ('9', '4', {'label': '-'}),
        ('10', '5', {'label': '-'})
    ]
    assert len(graph_edges) == len(expected_edges)
    for edge in expected_edges:
        assert edge in graph_edges

def test_calculate_node_features():
    file = f"{TEST_DIR}/framework1.txt"
    dep_graph = DependencyGraph()
    dep_graph.create_from_file(file)
    dep_graph.create_dependency_graph()

    graph_edges = list(dep_graph.graph.edges(data=True))
    (graph, rules_mapping, atoms_mapping) = reindex_nodes(dep_graph.graph)
    normalized_features = dep_graph.calculate_node_features(atoms_mapping)
    assert len(normalized_features.keys()) == 10
    assert len(list(normalized_features.values())[0]) == 2  

    
import networkx as nx
import numpy as np
import pickle as picklerick
import os 
from omegaconf import OmegaConf

from constraints.solvers.surrogate.surrogate import surrogate_reconstruction



def load_graph(filename):
    with open(filename, 'rb') as f:
        return picklerick.load(f)

def load_and_convert_graph(filename):
    graph = load_graph(filename)
    if isinstance(graph, nx.Graph):
        return graph
    else:
        raise ValueError("The loaded object is not a NetworkX graph")

def construct_model(cfg, problem_data):
    model = surrogate_reconstruction(cfg, ('classification', cfg['surrogate']['classifier_selection'], 'live_set_surrogate'), problem_data).rebuild_model()
    return model




if __name__ == "__main__":
    # Load the graph and the config file
    root_dir = "outputs/2024-12-05/09-08-32"
    extensions = "graph_backward_iterate_0.pickle"
    graph_file = os.path.join(root_dir, extensions)
    cfg_file = os.path.join(root_dir, ".hydra/config.yaml")
    graph = load_and_convert_graph(graph_file)
    cfg = OmegaConf.load(cfg_file)

    model_dictionary = {}
    # Load the surrogate model
    for node in graph.nodes:
        classifier_data = graph.nodes[node]['classifier_serialised']
        classifier = construct_model(cfg, classifier_data)   
        model_dictionary[node] = classifier


    print("Model dictionary loaded")
    




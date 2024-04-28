from abc import ABC
from functools import partial

from constraints.constructor import constraint_evaluator
from unit_evaluators.constructor import subproblem_unit_wrapper, network_simulator
from initialisation.methods import initialisation
from reconstruction.constructor import reconstruction
from visualisation.visualiser import visualiser
from surrogate.surrogate import surrogate
from solvers.constructor import solver_construction
from samplers.constructor import construct_deus_problem
from samplers.appproximators import calculate_box_outer_approximation
from samplers.space_filling import sobol_sample_design_space_nd
from samplers.utils import create_problem_description_deus
from deus import DEUS




def apply_nested_sampling(cfg, graph, mode:str="forward", max_devices=1):
    # TODO:
    # redefine the constraints to return direct constraint evaluations and not indicator values.
    # implement the methods to reconstruct
    
    # Create a list of nodes in the graph according to the precedence order
    nodes = nx.topological_sort(graph)


    if mode == "backward" or mode == "forward-backward":
        nodes = reversed(list(nodes))
    elif mode == "forward":
        nodes = list(nodes)
    else:
        raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward' or 'forward-backward'.")
        


    # Iterate over the nodes and apply nested sampling
    for node in nodes:
        logging.info(f'------- Characterising node {node} according to precedence order: {nodes} -------')
        # define model for deus
        model = ModelA(node, cfg, graph, mode=mode, notion_of_feasibility=cfg.notion_of_feasibility, evaluation_mode=cfg.evaluation_mode, max_devices=max_devices)
        # create problem sheet according to cfg
        problem_sheet = create_activity_form(cfg, model, graph, node) 
        # solve extended DS using NS
        live_set, deadpoints_all, model = run_deus_nested_sampling(
            problem_sheet, model, cfg, graph, node
        )
        # estimate box for bounds for DS downstream
        process_data_forward(cfg, graph, node, model, live_set)
        # train constraints for DS downstream using data now stored in the graph
        if mode == 'forward': surrogate_training_forward(cfg, graph, node)
        # classifier construction for current unit
        classifier_construction(cfg, graph, node)


    return graph


def surrogate_training_forward(cfg, graph, node):
    """
    Train the surrogate model for the forward pass.
    - only train the node if it has successors
    - train the node with the input data from the current node
    - store the trained model in the graph

    """

    # Check if the node has successors
    if graph.out_degree()[node] == 0:
        return
    
    for successor in graph.successors(node):
        # get the input data from the current node
        input_data = graph.edges[node, successor]["surrogate_training"]
        # train the model
        _, best_model, best_params, [x_scalar, y_scalar], query_model = hyperparameter_selection(cfg, input_data, cfg.k_fold.num_folds)
        # store the trained model in the graph
        graph.edges[node, successor]["forward_surrogate"] = query_model
        graph.nodes[node]['x_scalar'] = x_scalar
        graph.edges[node,successor]['y_scalar'] = y_scalar

    return



def process_data_forward(cfg, graph, node, model, live_set, notion_of_feasibility='positive'):
    """
    Process the data in the forward direction.

    Parameters:
    cfg (object): The configuration object with a 'classification_threshold' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.
    model (object): The model object with 'input_output_data' and 'classifier_data' attributes.
    live_set (jnp.array): The live set data.

    Returns:
    None
    """

    # Extract the input-output and classifier data from the model
    x_io = model.input_output_data.X
    y_io = model.input_output_data.y
    x_classifier = model.classifier_data.X
    y_classifier = model.classifier_data.y

    # Select a subset of the data based on the classifier
    if cfg.notion_of_feasibility == 'positive':
        select_cond = jnp.max(y_classifier, axis=1)  >= 0 
    else:
        select_cond = jnp.max(y_classifier, axis=1)  <= 0  

    selected_x, selected_y = x_io[select_cond.squeeze(), :], y_io[select_cond.squeeze(), :]

    # Apply the selected function to the y data and store forward evaluations on the graph
    for successor in graph.successors(node):
        # apply edge function to output data
        io_fn = graph.edges[node, successor]["edge_fn"]
        y_updated_io = io_fn(selected_y)
        if y_updated_io.ndim >2 : y_updated_io = y_updated_io.squeeze()
        # find box bounds on inputs
        graph.edges[node, successor][
            "input_data_bounds"
        ] = calculate_box_outer_approximation(y_updated_io, cfg)
        # store the forward evaluations on the graph for surrogate training
        y_in_node = io_fn(y_io)
        if y_in_node.ndim > 2: y_in_node= y_in_node.squeeze()
        if y_in_node.ndim < 2: y_in_node= y_in_node.reshape(-1,1)
        forward_evals = dataset_holder(X=x_io, y=y_in_node)
        graph.edges[node, successor]["surrogate_training"] = forward_evals   # TODO configure the option to train a local model here.

    # Store the classifier data and the live set data to the node
    graph.nodes[node]["classifier_training"] = model.classifier_data
    graph.nodes[node]["live_set_inner"] = live_set

    update_node_bounds_iplus1(graph, node, cfg)

    return

def update_node_bounds_iplus1(graph, node, cfg):
    """
    Update the bounds of the node in the graph for iterate i+1 based on the liveset of the node at iterate i.

    Parameters:
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    None
    """

    # Get the bounds of the node i
    new_bounds = calculate_box_outer_approximation(graph.nodes[node]["live_set_inner"], cfg)
    graph.nodes[node]['extendedDS_bounds'] = new_bounds

    return


def classifier_construction(cfg, graph, node):
    """
    Construct the classifier for the forward pass.

    Parameters:
    cfg (object): The configuration object with a 'classifier' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    classifier: The trained classifier. (-1 belongs to feasible region, 1 does not belong to feasible region)
    """

    

    return construct_coupling_constraint(graph, node, cfg)



class subproblem_model(ABC):
    def __init__(self, unit_index, cfg, G, mode, evaluation_mode, max_devices):
        self.function_evaluations = 0
        self.unit_index = unit_index
        self.cfg, self.G = cfg, G
        self.constraint_dictionary, self.constraint_args = G.nodes[unit_index]['constraints'], G.nodes[unit_index]['constraint_args']
        self.unit_forward_eval, self.input_args, self.n_design_args = subproblem_constructor(unit_index, G)
        self.input_output_data = None 
        self.classifier_data = None 
        self.mode = mode
        self.evaluation_mode = evaluation_mode
        self.max_devices = len(jax.devices('cpu'))
        print(f"Max devices: {self.max_devices}")
        print(f"{jax.devices('cpu')}")

    def determine_batches(self, data, batch_size):
        """ Method to determine the number of batches"""
        n_batches = data.shape[0] // batch_size
        if data.shape[0] % batch_size != 0:
            n_batches += 1
        return n_batches
    
    def evaluate_subproblem_batch(self, data, batch_size, p):
        """ Method to evaluate the subproblem in batches"""
        n_batches = self.determine_batches(data, batch_size)
        constraints = []
        for i in range(n_batches):
            batch = data[i*batch_size:(i+1)*batch_size,:]
            constraints.append(self.subproblem_constraint_evals(batch, p))

        return np.vstack(constraints)
        
    def subproblem_constraint_evals(self, d, p):
        
        # unpack design args
        if self.input_args == None:
            input_args = None
            design_args = jnp.array(d[:,:])
        else:
            input_args = jnp.array(d[:,self.n_design_args:]) # take the rest of the args
            design_args = jnp.array(d[:,:self.n_design_args]) # take the first n args

        # unpack dynamics terms
        forward_eval = self.unit_forward_eval
        # unit forward pass
        outputs = forward_eval(self.cfg, design_args, input_args, jnp.array(p), self.unit_index) # NOTE it is assumed that we vmap evaluations here. config, design_args, input_args, uncertain_params, unit_index
        # evaluate process constraints # NOTE this could be written to just operate on graph
        constraint_evals = constraint_evaluation_fn(self.constraint_dictionary, outputs, self.cfg, self.constraint_args, self.cfg.case_study)
        # evaluate feasibility upstream
        upstream_feasibility_evals = upstream_feasibility_handler(input_args, self.cfg, self.unit_index, self.G, mode=self.mode, notion_of_feasibility=self.notion_of_feasibility, evaluation_mode=self.evaluation_mode)
        upstream_processed = constraint_dummy_check(upstream_feasibility_evals)
        # evaluate feasibility downstream
        downstream_feasibility_evals = downstream_feasibility_handler(outputs, self.cfg, self.unit_index, self.G, mode=self.mode, notion_of_feasibility=self.notion_of_feasibility, evaluation_mode=self.evaluation_mode)
        downstream_processed = constraint_dummy_check(downstream_feasibility_evals)

        # update input output data for forward surrogate model
        self.input_output_data = update_data(self.input_output_data, d, outputs)  # updating dataset for surrogate model of forward unit evaluation

        # constraint evals
        constraint_s = [cons for cons in constraint_evals.values()]
        constraint_s += [upstream_processed, downstream_processed]

        return np.hstack([c.reshape(-1,1) for c in constraint_s if not (c is None)])


    def s(self, d, p):
        # evaluate feasibility and then update classifier data and number of function evaluations
        g = self.evaluate_subproblem_batch(d, self.max_devices, p.reshape(1,1,-1))
        self.classifier_data = update_data(self.classifier_data, d, g)  # updating dataset for surrogate model of forward unit evaluation
        self.function_evaluations += g.shape[0]
        #logging.info('candidate evaluted')

        return [g[i,:].reshape(1,-1) for i in range(g.shape[0])]
        

    def get_constraints(self, d, p):
        return self.s(d, p)



def subproblem_constructor(unit: int, G: nx.DiGraph):
    """ Method to construct subproblem for each unit in the network"""

    if unit not in G.nodes:
        raise ValueError(f"Unit: {unit} is not a node in the graph G")

    if G.in_degree()[unit] == 0:
        input_args, n_design_args  = None, G.nodes[unit]['n_design_args']
        return G.nodes[unit]['forward_evaluator'], input_args, n_design_args
    elif  G.in_degree()[unit] > 0:
        input_args, n_design_args = G.nodes[unit]['n_design_args'], G.nodes[unit]['n_design_args'] # use hashing to return correct input args and design args
        return G.nodes[unit]['forward_evaluator'], input_args, n_design_args
    else:
        raise ValueError(f"Unit: {unit} not recognised")
    
    
def update_data(data, X, y):
    """ Method to update the data holder with new data"""
    if y.ndim >2: y = y.squeeze()
    if X.ndim >2: X = X.squeeze()
    if data is None:
        data = gpx.Dataset(X=X, y=y)
    else:
        data = data.__add__(gpx.Dataset(X=X, y=y))

    return data

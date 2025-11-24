import jax.numpy as jnp
import numpy as np
from scipy.stats import beta
from jax import jit, pmap, devices, lax
from functools import partial

from mu_F.solvers.functions import generate_initial_guess, multi_start_solve_bounds_nonlinear_program
from mu_F.constraints.utils import standardise_inputs, standardise_model_decisions, mask_classifier, get_successor_inputs
from mu_F.solvers.utilities import determine_batches, create_batches
       
def assess_feasibility(feasibility, input):
    """
    Assesses the feasibility of the input
    """
    if feasibility == 'positive':
        return input >= 0
    elif feasibility == 'negative':
        return input <= 0
    else:
        raise ValueError("Invalid notion of feasibility.")
    

""" ---- JaxOpt solver evaluation methods (written as pure functions not classes) --- """


def shaping_function(x, cfg):
    """
    Shaping function
    """
    if cfg.samplers.notion_of_feasibility == 'positive':
        return -x
    elif cfg.samplers.notion_of_feasibility == 'negative':
        return x


def construct_solver(objective_func, bounds, tol):
    bounds = bounds
    objective_func = objective_func
    bounds = bounds
    solver = partial(multi_start_solve_bounds_nonlinear_program, objective_func=objective_func, bounds_=(bounds[0], bounds[1]), tol=tol)
    return solver   

def initial_guess(cfg, bounds):
    n_d = len(bounds[0])
    return generate_initial_guess(cfg.n_starts, n_d, bounds)

def solve(solver, initial_guesses):
    obj_r, e  = [], []

    for solve, init in zip(solver, initial_guesses):
        objective, error = solve(init)
        obj_r.append(objective)
        e.append(error)
    
    return {'objective': jnp.array(obj_r), 'error': jnp.array(e)}
    

def load_solver(objective_func, bounds):
    """
    Loads the solver
    """
    return construct_solver(objective_func, bounds)


def prepare_backward_problem(outputs, graph, node, cfg):
    """
    Prepares the forward constraints surrogates and decision variables
    - ouptuts from a nodes unit functions are inputs to the next unit

    """
    if node is None:
        return None, None
    else: 
        # TODO make sure that this is not going to throw errors in tracing.
        backward_bounds = {succ: None for succ in graph.successors(node)}
        backward_objective = {succ: None for succ in graph.successors(node)}

        # get the outputs from the successors of the node
        succ_inputs = get_successor_inputs(graph, node, outputs)

        for succ in graph.successors(node):

            n_d  = graph.nodes[succ]['n_design_args']
            input_indices = np.copy(np.array([n_d + input_ for input_ in graph.edges[node, succ]['input_indices']]))
            aux_indices = np.copy(np.array([input_ for input_ in graph.edges[node, succ]['auxiliary_indices']]))
        
            
            # standardisation of outputs if required
            if cfg.solvers.standardised: succ_inputs[succ] = succ_inputs[succ].at[:].set(standardise_inputs(graph, succ_inputs[succ], succ, jnp.hstack([input_indices, aux_indices]).astype(int)))
            
            # load the standardised bounds
            decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
            ndim = graph.nodes[succ]['n_design_args'] + graph.nodes[succ]['n_input_args'] + graph.graph['n_aux_args']
            decision_indices = jnp.delete(jnp.arange(ndim), np.hstack([input_indices, aux_indices]).astype(int))  # indices of the decision variables
            # get the decision bounds
            if cfg.solvers.standardised: decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
            
            decision_bounds = [jnp.delete(bound, np.hstack([input_indices,aux_indices]).astype(int), axis=1) for bound in decision_bounds]
            backward_bounds[succ] = [decision_bounds.copy() for i in range(succ_inputs[succ].shape[0])]

            # load the forward objective
            classifier = graph.nodes[succ]["classifier"]
            wrapper_classifier = mask_classifier(classifier, ndim, input_indices, aux_indices)
            backward_objective[succ] = [jit(partial(lambda x,y: wrapper_classifier(x,y).squeeze(), y=succ_inputs[succ][i].reshape(1,-1))) for i in range(succ_inputs[succ].shape[0])]

        # return the forward surrogates and decision bounds
        return backward_objective, backward_bounds

def prepare_global_problem(inputs, aux, graph, cfg):
    """
    Prepares the global problem defined for handling nuisance parameters in the reconstruction 
        - loads the objective function and bounds from the graph by
            1: loads the classifier from the graph 
            2: loads the bounds from the graph
            3: loads the fixed indices from the graph
            4: standardises the inputs and decisions if required
            5: masks the classifier to only use the decision variables
            6: prepares the global problem for the solver
    """
    n_d     = graph.graph['n_design_args'] # number of design variables in the successors of the root node
    n_aux   = graph.graph['n_aux_args']

    # get the fixed indices and auxiliary indices
    dec_ind = np.array(graph.graph['post_process_decision_indices'])
    total_ind = np.arange(n_d + n_aux)
    fix_ind = np.delete(total_ind, dec_ind).astype(int)  # indices of the fixed decision variables

    # introduce bounds 
    lb =     jnp.hstack([jnp.array(bound[0]).reshape(-1,) for bound in graph.graph['bounds'] if bound[0] != 'None'])
    ub =     jnp.hstack([jnp.array(bound[1]).reshape(-1,) for bound in graph.graph['bounds'] if bound[1] != 'None'])
    bounds = [lb, ub]
    
    # standardise the inputs and decisions if required
    if cfg.solvers.standardised:
        inputs = standardise_inputs(graph, inputs, None, jnp.hstack([fix_ind]).astype(int))
        bounds = standardise_model_decisions(graph, bounds, None)

    # mask the classifier to only use the decision variables
    classifier = mask_classifier(graph.graph['post_process_lower_classifier'], n_d + n_aux, fix_ind, np.empty((0,)).astype(int))

    # prepare the objective function # NOTE this should be a maximization problem -> therefore negative values of the objective indicate constraint violations.
    objective_func = partial(lambda x, y: -classifier(x, y).squeeze(), y=inputs.reshape(1,-1))

    # prepare the bounds
    bounds = [jnp.delete(bounds[0], fix_ind), jnp.delete(bounds[1], fix_ind)]
    

    return objective_func, bounds



def evaluate(outputs, aux, graph, node, cfg):
    """
    Evaluates the constraints.
    Handles both graph-wide and node-local (backward) problems.
    Amenable to jax pmap by using jax.lax.cond for control flow.
    """

    evaluate_method = solve

    def graph_wide_branch(args):
        (outputs, aux) = args
        if outputs.ndim < 2: outputs = outputs.reshape(-1, 1)
        if aux.ndim < 2: aux = aux.reshape(-1, 1)
        objective, bounds = prepare_global_problem(outputs, aux, graph, cfg)
        solver = construct_solver(objective, bounds, tol=cfg.solvers.post.jax_opt_options.error_tol)
        initial_guesses = initial_guess(cfg.solvers.backward_coupling, bounds)
        result = evaluate_method([solver], [initial_guesses])
        fn_evaluations = result['objective'].reshape(-1, 1)
        return -shaping_function(fn_evaluations, cfg) # maximisation problem

    def node_local_branch(args):
        (outputs, aux) = args
        # note that auxiliary variables are assumed global and propagated through the graph constituent functions
        objective, bounds = prepare_backward_problem(outputs, graph, node, cfg)
        # enabling tracing
        if objective is None or bounds is None:
            return jnp.zeros((outputs.shape[0], 1))
        # function body
        else: 
            succ_fn_evaluations = {}
            for succ in graph.successors(node):
                backward_solver = [
                    construct_solver(objective[succ][i], bounds[succ][i], tol=cfg.solvers.backward_coupling.jax_opt_options.error_tol)
                    for i in range(outputs.shape[0])
                ]
                initial_guesses = [
                    initial_guess(cfg.solvers.backward_coupling, bounds[succ][i])
                    for i in range(outputs.shape[0])
                ]
                succ_fn_evaluations[succ] = evaluate_method(backward_solver, initial_guesses)
            fn_evaluations = [
                succ_fn_evaluations[succ]['objective'].reshape(-1, 1)
                for succ in graph.successors(node)
            ]
            return shaping_function(jnp.hstack(fn_evaluations), cfg)

    is_graph_wide = graph.graph["solve_post_processing_problem"]
    return lax.cond(
        is_graph_wide,
        false_fun = node_local_branch,
        true_fun = graph_wide_branch,
        operand = (outputs, aux)
    )


def jax_pmap_evaluator(outputs, aux, cfg, graph, node):
    """
    p-map constraint evaluation call - called by backward_surrogate_pmap_batch_evaluator
    """
    constraint_evaluator = partial(evaluate, graph=graph, node=node, cfg=cfg)

    return constraint_evaluator(outputs, aux)


def backward_surrogate_pmap_batch_evaluator(outputs, aux, cfg, graph, node):
    """
    Evaluates the constraints on a batch using jax-pmap - called by the backward_constraint_evaluator
    """
    feasibility_call = partial(jax_pmap_evaluator, cfg=cfg, graph=graph, node=node)
    
    return pmap(feasibility_call, in_axes=(0,0), out_axes=0, devices=[device for i, device in enumerate(devices('cpu')) if i<outputs.shape[0]])(outputs, aux)  #, axis_name='i'
   
 

def backward_constraint_evaluator(outputs, aux, cfg, graph, node, pool):
    """
    Evaluates the constraints using jax-pmap - this is what should be called

    Syntax: 
        call: method_(outputs, cfg, graph, node, pool)

    """
    max_devices = cfg.max_devices
    batch_sizes, remainder = determine_batches(outputs.shape[0], max_devices)
    # get batches of outputs
    output_batches = create_batches(batch_sizes, outputs)
    aux_batches = create_batches(batch_sizes, jnp.repeat(jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1))
    # evaluate the constraints
    results = []
    for i, (output_batch, aux_batch) in enumerate(zip(output_batches, aux_batches)):
        results.append(backward_surrogate_pmap_batch_evaluator(output_batch, aux_batch, cfg, graph, node))
    # concatenate the results

    del output_batches, aux_batches, batch_sizes

    return jnp.vstack(results)

    